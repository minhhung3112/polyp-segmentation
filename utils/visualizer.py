"""
Visualization & Grad-CAM Utilities for Polyp Segmentation (A* Q1 Publication Standards).

Generates:
1. Pure individual images for paper figures (NO titles, NO borders, NO axes):
   - input/      : Original raw colonoscopy frame
   - gt/         : Ground truth binary mask
   - pred/       : Predicted probability mask
   - gradcam/    : Grad-CAM attention heatmap (jet colormap overlaid on input)
   - error/      : Diagnostic Error Map (Green=TP, Red=FP, Blue=FN)
   - overlay/    : Translucent green mask overlay on colonoscopy frame
2. Multi-dataset comparative bar chart across benchmark metrics
3. 4-Panel diagnostic comparison figure
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for medical segmentation models.
    Hooks into intermediate feature maps and gradients to visualize attention focus.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        if self.target_layer is None:
            self._auto_find_target_layer()

        if self.target_layer is not None:
            self._register_hooks()

    def _auto_find_target_layer(self):
        # Look for suitable deep convolutional layer in the backbone or encoder
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
                break

    def _save_gradient(self, grad):
        self.gradients = grad

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate(self, img_tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap [0, 1] for input tensor.
        """
        self.model.zero_grad()
        img_tensor = img_tensor.clone().requires_grad_(True)
        
        output = self.model(img_tensor)
        pred = output[-1] if isinstance(output, (tuple, list)) else output
        
        # Target polyp prediction score
        score = pred.sigmoid().sum()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback to feature map magnitude if hook didn't capture
            return np.zeros(target_size, dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


class Visualizer:
    """Handles all visual reporting for polyp segmentation models."""

    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_paper_figures(
        self,
        image: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        cam_heatmap: np.ndarray,
        sample_name: str,
        base_dir: str
    ):
        """
        Export pure, borderless, textless images for paper subfigures:
            - input/   : Original raw colonoscopy image
            - gt/      : Ground truth binary mask (0 and 255)
            - pred/    : Model prediction mask
            - gradcam/ : Grad-CAM attention heatmap overlaid on input image
            - error/   : Diagnostic Error Map (Green=TP, Red=FP, Blue=FN)
            - overlay/ : Translucent green mask overlay on colonoscopy frame
        """
        if image.max() <= 1.0:
            img_u8 = (image * 255.0).astype(np.uint8)
        else:
            img_u8 = image.astype(np.uint8)

        gt_b = (gt > 0.5).astype(np.float32)
        gt_u8 = (gt_b * 255.0).astype(np.uint8)

        pred_b = (pred > 0.5).astype(np.float32)
        pred_u8 = (np.clip(pred, 0, 1) * 255.0).astype(np.uint8)

        # 1. Error map
        h, w = gt_b.shape[:2]
        error_map = np.zeros((h, w, 3), dtype=np.uint8)
        tp = (pred_b == 1) & (gt_b == 1)
        fp = (pred_b == 1) & (gt_b == 0)
        fn = (pred_b == 0) & (gt_b == 1)
        error_map[tp] = [0, 204, 0]    # Green: TP
        error_map[fp] = [230, 25, 25]  # Red: FP
        error_map[fn] = [25, 102, 230] # Blue: FN

        # 2. Translucent Green Overlay
        alpha = 0.45
        overlay = img_u8.astype(np.float32).copy()
        overlay[..., 1] = np.where(pred_b > 0.5, (1 - alpha) * overlay[..., 1] + alpha * 240, overlay[..., 1])
        overlay[..., 0] = np.where(pred_b > 0.5, (1 - alpha) * overlay[..., 0], overlay[..., 0])
        overlay[..., 2] = np.where(pred_b > 0.5, (1 - alpha) * overlay[..., 2], overlay[..., 2])
        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)

        # 3. Grad-CAM Jet Colormap Overlay
        cam_u8 = (np.clip(cam_heatmap, 0, 1) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        gradcam_overlay = (0.5 * img_u8.astype(np.float32) + 0.5 * heatmap_color.astype(np.float32)).astype(np.uint8)

        # Subfolders
        folders = ['input', 'gt', 'pred', 'gradcam', 'error', 'overlay']
        for f in folders:
            os.makedirs(os.path.join(base_dir, f), exist_ok=True)

        # Save individual images without text, border, or axes
        Image.fromarray(img_u8).save(os.path.join(base_dir, 'input', f"{sample_name}.png"))
        Image.fromarray(gt_u8).save(os.path.join(base_dir, 'gt', f"{sample_name}.png"))
        Image.fromarray(pred_u8).save(os.path.join(base_dir, 'pred', f"{sample_name}.png"))
        Image.fromarray(gradcam_overlay).save(os.path.join(base_dir, 'gradcam', f"{sample_name}.png"))
        Image.fromarray(error_map).save(os.path.join(base_dir, 'error', f"{sample_name}.png"))
        Image.fromarray(overlay_u8).save(os.path.join(base_dir, 'overlay', f"{sample_name}.png"))

    def plot_sample_grid(
        self,
        image: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        cam_heatmap: np.ndarray,
        sample_name: str,
        save_path: str = None
    ):
        """Render 5-panel figure: [Input | GT | Prediction | Grad-CAM | Error Map]."""
        if image.max() > 1.0:
            image = image / 255.0

        gt_b = (gt > 0.5).astype(np.float32)
        pred_b = (pred > 0.5).astype(np.float32)

        # Error map
        h, w = gt_b.shape[:2]
        error_map = np.zeros((h, w, 3), dtype=np.float32)
        tp = (pred_b == 1) & (gt_b == 1)
        fp = (pred_b == 1) & (gt_b == 0)
        fn = (pred_b == 0) & (gt_b == 1)
        error_map[tp] = [0.0, 0.8, 0.0]
        error_map[fp] = [0.9, 0.1, 0.1]
        error_map[fn] = [0.1, 0.4, 0.9]

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        axes[0].imshow(image)
        axes[0].set_title("Input Frame", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(gt_b, cmap='gray')
        axes[1].set_title("Ground Truth", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(pred, cmap='magma', vmin=0, vmax=1)
        axes[2].set_title("Prediction", fontsize=11, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(image)
        axes[3].imshow(cam_heatmap, cmap='jet', alpha=0.5)
        axes[3].set_title("Grad-CAM Heatmap", fontsize=11, fontweight='bold')
        axes[3].axis('off')

        axes[4].imshow(error_map)
        axes[4].set_title("Diagnostic Error Map", fontsize=11, fontweight='bold')
        axes[4].axis('off')

        plt.suptitle(f"Sample: {sample_name}", fontsize=13, y=1.02)
        plt.tight_layout()

        target_file = save_path or os.path.join(self.save_dir, f"grid_{sample_name}.png")
        os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
        plt.savefig(target_file, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_metrics_bar(self, results_dict: dict, save_path: str = None):
        """Plot comparative bar chart across evaluated datasets."""
        datasets = [k for k in results_dict.keys() if k != 'Mean']
        if not datasets:
            return

        metrics = ['mDice', 'mIoU', 'wFb', 'Sm', 'Em']
        x = np.arange(len(datasets))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, m in enumerate(metrics):
            vals = [results_dict[d].get(m, 0.0) for d in datasets]
            ax.bar(x + idx * width, vals, width, label=m)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Dataset Performance Profile', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=11)

        plt.tight_layout()
        target_file = save_path or os.path.join(self.save_dir, "metrics_bar_chart.png")
        os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
        plt.savefig(target_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Saved metrics bar chart to '{target_file}'.")
