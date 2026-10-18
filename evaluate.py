"""
Offline / Standalone Evaluation & Edge AI Benchmark for Polyp Segmentation.

Computes:
1. Accuracy Metrics: mDice, mIoU, wFb, Sm, Em, MAE directly from saved prediction masks.
2. Edge AI & Efficiency Metrics: Parameters (M), FLOPs (G), Latency (GPU & CPU ms), FPS, Memory.
"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch

from utils.metrics import MetricCalculator
from utils.setup_dataset import setup_dataset
from utils.edge_profiler import EdgeProfiler
from models import get_model, list_models

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def main():
    parser = argparse.ArgumentParser(description="Offline Metric Evaluation & Edge AI Benchmark")
    parser.add_argument('--model', type=str, required=True, help='Model name corresponding to folder in --pred_root')
    parser.add_argument('--pred_root', type=str, default='./results', help='Root directory containing predicted masks')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root directory')
    parser.add_argument('--testsize', type=int, default=352, help='Input resolution for hardware profiling')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to weights file (.pth)')
    parser.add_argument('--profile', action='store_true', default=True, help='Profile Edge AI hardware metrics (Params, FLOPs, Latency, FPS)')
    parser.add_argument('--no_profile', dest='profile', action='store_false', help='Disable Edge AI hardware profiling')
    args = parser.parse_args()

    # 1. Edge AI Hardware Profiling if model is available in registry
    results_model_dir = os.path.join(args.pred_root, args.model)
    os.makedirs(results_model_dir, exist_ok=True)

    if args.profile and args.model.lower() in [m.lower() for m in list_models()]:
        print(f"\n[Evaluate] Profiling Edge AI & Hardware efficiency for '{args.model}'...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = get_model(args.model)
            profiler = EdgeProfiler(
                model=model,
                testsize=args.testsize,
                device=device,
                weights_path=args.weights
            )
            eff_results = profiler.profile(measure_cpu=True)
            EdgeProfiler.print_table(eff_results, model_name=args.model)
            EdgeProfiler.save_csv(eff_results, model_name=args.model, path=os.path.join(results_model_dir, 'efficiency.csv'))
            EdgeProfiler.save_latex(eff_results, model_name=args.model, path=os.path.join(results_model_dir, 'efficiency.tex'))
        except Exception as e:
            print(f"[Evaluate] Warning: Failed to profile hardware efficiency ({e})")

    # 2. Accuracy Benchmark across 5 test sets from saved prediction masks
    data_root = setup_dataset(args.data_root)
    test_root = os.path.join(data_root, 'TestDataset')

    test_sets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    results = {}

    print(f"\n[Evaluate] Computing segmentation accuracy metrics for '{args.model}' from '{args.pred_root}'...")

    for ds_name in test_sets:
        pred_dir = os.path.join(args.pred_root, args.model, ds_name)
        gt_dir = os.path.join(test_root, ds_name, 'masks')

        if not os.path.exists(pred_dir):
            print(f"[Evaluate] Skipping '{ds_name}': predictions folder '{pred_dir}' not found.")
            continue
        if not os.path.exists(gt_dir):
            print(f"[Evaluate] Skipping '{ds_name}': GT folder '{gt_dir}' not found.")
            continue

        pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg'))])
        if not pred_files:
            continue

        calc = MetricCalculator()
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)}

        for p_name in tqdm(pred_files, desc=f"Scoring {ds_name}"):
            stem = os.path.splitext(p_name)[0]
            if stem not in gt_files:
                continue

            pred_p = os.path.join(pred_dir, p_name)
            gt_p = os.path.join(gt_dir, gt_files[stem])

            pred_img = Image.open(pred_p).convert('L')
            gt_img = Image.open(gt_p).convert('L')

            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.BILINEAR)

            pred_arr = np.asarray(pred_img, dtype=np.float32) / 255.0
            gt_arr = np.asarray(gt_img, dtype=np.float32) / 255.0

            calc.update(pred_arr, gt_arr)

        res = calc.get_results()
        results[ds_name] = res

    if results:
        # Calculate Mean row
        metric_keys = list(next(iter(results.values())).keys())
        mean_row = {}
        for k in metric_keys:
            mean_row[k] = float(np.mean([results[ds][k] for ds in results]))
        results['Mean'] = mean_row

        df = pd.DataFrame(results).T
        cols = ['mDice', 'mIoU', 'Recall', 'Precision', 'Specificity', 'HD95', 'wFb', 'Sm', 'Em', 'MAE']
        display_cols = [c for c in cols if c in df.columns]
        df_disp = df[display_cols]

        print("\n" + "=" * 70)
        print(f"OFFLINE ACCURACY BENCHMARK RESULTS: {args.model.upper()}")
        print("=" * 70)
        if HAS_TABULATE:
            print(tabulate(df_disp, headers='keys', tablefmt='psql', floatfmt=".4f"))
        else:
            print(df_disp.to_string())
        print("=" * 70 + "\n")

        # Save CSV and LaTeX
        out_csv = os.path.join(results_model_dir, 'metrics.csv')
        out_tex = os.path.join(results_model_dir, 'metrics.tex')
        df.to_csv(out_csv)
        with open(out_tex, 'w', encoding='utf-8') as f:
            f.write(df_disp.to_latex(float_format="%.4f"))
        print(f"[Evaluate] Results saved to '{out_csv}' and '{out_tex}'.")
    else:
        print("[Evaluate] No matching predictions found to evaluate.")


if __name__ == '__main__':
    main()
