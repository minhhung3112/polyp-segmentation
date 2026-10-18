"""
CLI Visualization Script for Polyp Segmentation Models (A* Q1 Publication Suite).

Generates:
1. Multi-dataset pure images for paper figures (NO titles, NO borders, NO labels):
   - visualizations/{model}/paper_figures/{dataset}/input/
   - visualizations/{model}/paper_figures/{dataset}/gt/
   - visualizations/{model}/paper_figures/{dataset}/pred/
   - visualizations/{model}/paper_figures/{dataset}/gradcam/
   - visualizations/{model}/paper_figures/{dataset}/error/
   - visualizations/{model}/paper_figures/{dataset}/overlay/
2. 5-Panel diagnostic comparison figures
3. Comparative performance bar charts
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils.checkpoint import Checkpointer
from utils.visualizer import Visualizer, GradCAM
from utils.dataloader import TestDataset
from utils.setup_dataset import setup_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Paper Figures & Grad-CAM for Polyp Segmentation")
    parser.add_argument('--model', type=str, default='pranet', help='Model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file (.pth)')
    parser.add_argument('--dataset', type=str, default='all', help='Test dataset name or "all" to export across all 5 test sets')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of sample images per dataset')
    parser.add_argument('--testsize', type=int, default=352, help='Input resolution')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output visualization directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = args.output_dir or os.path.join('./visualizations', args.model)
    viz = Visualizer(save_dir=output_dir)

    # 1. Setup dataset
    data_root = setup_dataset(args.data_root)

    # 2. Load model
    print(f"[Visualize] Initializing model '{args.model}' on {device}...")
    model = get_model(args.model)
    Checkpointer.load(model, args.weights, device=device)
    model.to(device)
    model.eval()

    gradcam = GradCAM(model)

    # 3. Determine datasets to visualize
    all_datasets = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    if args.dataset.lower() == 'all':
        target_datasets = all_datasets
    else:
        target_datasets = [args.dataset]

    print(f"[Visualize] Exporting publication figures across datasets: {target_datasets}")

    for dset in target_datasets:
        img_dir = os.path.join(data_root, 'TestDataset', dset, 'images')
        gt_dir = os.path.join(data_root, 'TestDataset', dset, 'masks')

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            print(f"[Visualize] Skipping '{dset}': directory not found.")
            continue

        test_loader = TestDataset(img_dir, gt_dir, testsize=args.testsize)
        num_to_viz = min(args.num_samples, test_loader.size)
        print(f"[Visualize] Processing {num_to_viz} samples from '{dset}'...")

        paper_dir = os.path.join(output_dir, 'paper_figures', dset)

        for idx in range(num_to_viz):
            img_tensor, gt_pil, name = test_loader.load_data()
            gt_np = np.asarray(gt_pil, dtype=np.float32)
            if gt_np.max() > 0:
                gt_np /= gt_np.max()

            orig_h, orig_w = gt_np.shape
            img_path = os.path.join(img_dir, name)
            if not os.path.exists(img_path):
                img_path = os.path.splitext(img_path)[0] + '.jpg'
            orig_img_pil = Image.open(img_path).convert('RGB')
            orig_img_np = np.asarray(orig_img_pil, dtype=np.float32)

            img_t = img_tensor.to(device)

            # Grad-CAM heatmap
            cam_map = gradcam.generate(img_t, target_size=(orig_h, orig_w))

            with torch.no_grad():
                outputs = model(img_t)
                pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                pred_np = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

            stem = os.path.splitext(name)[0]

            # Export pure images for paper (NO text, NO border, pure resolution)
            viz.save_paper_figures(
                orig_img_np, gt_np, pred_norm, cam_map,
                sample_name=stem, base_dir=paper_dir
            )

            # Also save 5-panel diagnostic grid (with Grad-CAM column)
            viz.plot_sample_grid(
                orig_img_np, gt_np, pred_norm, cam_map, sample_name=f"{dset}_{stem}",
                save_path=os.path.join(output_dir, f"grid_{dset}_{stem}.png")
            )

    gradcam.remove_hooks()

    # 4. If metrics CSV exists, render bar chart
    csv_path = os.path.join('./results', args.model, 'metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        results_dict = df.to_dict(orient='index')
        viz.plot_metrics_bar(results_dict, save_path=os.path.join(output_dir, f"{args.model}_metrics_bar.png"))

    print(f"\n[Visualize] Completed! All paper subfigures and Grad-CAM maps exported to '{output_dir}'.")


if __name__ == '__main__':
    main()
