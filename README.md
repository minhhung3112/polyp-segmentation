# End-to-End Polyp Segmentation Framework (PraNet Benchmark & A* Q1 Standards)

An extensible, production-grade PyTorch framework for automated polyp segmentation in colonoscopy imaging, strictly adhering to the methodology, architecture, and training regime of **PraNet (MICCAI 2020)** and benchmarking standards across top-tier venues (MICCAI, CVPR, ICCV, IEEE TMI, IEEE JBHI).

---

## 🌟 Key Highlights

- **Plug-and-Play Model & Loss Registry**: Automatic module discovery via `@register_model('name')` and `@register_loss('name')`. Simply drop a `.py` file into `models/` or `loss/` and run without modifying any central imports.
- **One-Command Colab Execution**: Automatically fetches and extracts the polyp benchmark dataset from [HuggingFace (`doantrongthai/Polyp_Segmentation`)](https://huggingface.co/datasets/doantrongthai/Polyp_Segmentation).
- **Exact PraNet Reproduction**:
  - Backbone: Self-contained **Res2Net-50 v1b (26w×4s)** with ImageNet pretrained initialization.
  - Multi-scale training: $[0.75, 1.0, 1.25]$ on each batch.
  - Deep supervision: Simultaneous supervision on all 4 decoder scales (`lateral_map_5`, `4`, `3`, `2`).
  - Loss formulation: Exact **Structure Loss** combining area-weighted BCE and weighted IoU.
  - Learning rate schedule: Step decay factor $0.1$ every $50$ epochs with gradient clipping ($0.5$).
- **A* Q1 Metric Suite**:
  - Mean Dice Similarity Coefficient ($\text{mDice}$)
  - Mean Intersection-over-Union ($\text{mIoU}$)
  - Weighted F-measure ($F_\beta^w$, Margolin et al., CVPR 2014)
  - Structure-measure ($S_\alpha$, Fan et al., ICCV 2017)
  - Enhanced-alignment measure ($E_\xi$, Fan et al., IJCAI 2018)
  - Mean Absolute Error ($\text{MAE}$)
  - Precision, Recall, Specificity
- **Full Benchmark Evaluation**: Evaluates across all 5 standard clinical benchmarks:
  1. **Kvasir-SEG** (100 test images)
  2. **CVC-ClinicDB** (62 test images)
  3. **CVC-ColonDB** (380 test images)
  4. **CVC-300** (60 test images)
  5. **ETIS-LaribPolypDB** (196 test images)
- **Publication Visual Diagnostics**:
  - 4-panel comparison: `[Input Frame | Ground Truth | Prediction | Diagnostic Error Map]` (Green=TP, Red=FP, Blue=FN).
  - Translucent green contour overlay on raw endoscopic frames.
  - Cross-dataset performance bar charts and training loss/validation curves.

---

## 🚀 Quick Start (Google Colab / Local)

### 1. Installation
```bash
git clone <your-repo-url>
cd PolypSeg_Project
pip install -r requirements.txt
```

### 2. Training
Single entry point with model, loss, and seed selection. All hyperparameter settings are hardcoded to match the PraNet paper:
```bash
!python train.py --model pranet --loss structure_loss --seed 42
```
*Note: The dataset is automatically downloaded and extracted on the first run.*

### 3. Evaluation on Checkpoint Weights
Evaluate on all 5 test sets and print PSQL table, save CSV, and generate LaTeX code:
```bash
!python test.py --model pranet --weights snapshots/pranet/best.pth
```

### 4. Standalone / Offline Evaluation (Model-free)
Directly scores saved predicted mask PNGs in `results/pranet/` against ground truth masks:
```bash
!python evaluate.py --model pranet
```

### 5. Visualization Generation
Generates diagnostic error maps, overlays, and performance bar charts:
```bash
!python visualize.py --model pranet --weights snapshots/pranet/best.pth --dataset Kvasir --num_samples 8
```

---

## 📁 Repository Structure

```
PolypSeg_Project/
├── train.py                  # Single training entry point
├── test.py                   # Checkpoint evaluation on all 5 test sets
├── evaluate.py               # Standalone evaluation from saved mask images
├── visualize.py              # Visual diagnostic generation script
├── requirements.txt          # Library dependencies (unpinned for colab compatibility)
├── README.md                 # Full documentation
├── models/
│   ├── __init__.py           # Dynamic auto-import & model registry
│   ├── pranet.py             # PraNet (MICCAI 2020) [30.50M]
│   ├── polyp_pvt.py          # Polyp-PVT (CAAI 2023) [25.11M]
│   ├── hardnet_mseg.py       # HarDNet-MSEG (CMPB 2021) [17.42M]
│   ├── sanet.py              # SANet (MICCAI 2021) [23.90M]
│   ├── unet.py               # U-Net (MICCAI 2015) [31.04M]
│   ├── unet_plus_plus.py     # UNet++ (DLMIA 2018) [9.16M]
│   ├── enet.py               # ENet (arXiv 2016) [0.40M]
│   ├── edanet.py             # EDANet (IEEE TMM 2019) [0.61M]
│   ├── espnetv2.py           # ESPNetv2 (CVPR 2019) [0.79M]
│   ├── malunet.py            # MALUNet (BIBM 2022) [0.18M]
│   ├── fast_scnn.py          # Fast-SCNN (BMVC 2019) [1.14M]
│   ├── cmunext.py            # CMUNeXt base (arXiv 2024) [2.33M]
│   ├── lgps.py               # LGPS (2024) [2.69M]
│   ├── emcad.py              # EMCAD-B0 (CVPR 2024) [4.36M]
│   ├── gcascade.py           # G-CASCADE-B0 (WACV 2024) [4.64M]
│   ├── meganet.py            # MEGANet (2024) [44.19M]
│   └── _backbone/
│       ├── __init__.py
│       └── res2net.py        # Self-contained Res2Net-50 v1b (26w_4s)
├── loss/
│   ├── __init__.py           # Dynamic auto-import & loss registry
│   ├── structure_loss.py     # Weighted BCE + Weighted IoU (PraNet, Polyp-PVT, HarDNet)
│   ├── bce.py                # Binary Cross-Entropy with Logits
│   ├── dice.py               # Soft Dice Loss
│   ├── bce_dice.py           # Combined BCE + Dice loss
│   ├── focal_loss.py         # Focal Loss (ICCV 2017)
│   ├── focal_dice.py         # Focal + Dice Compound Loss (SANet, ColonSegNet)
│   ├── iou_loss.py           # Soft Jaccard / IoU Loss
│   ├── tversky_loss.py       # Tversky Loss with asymmetric FP/FN penalties (MLMI 2017)
│   ├── focal_tversky_loss.py # Focal Tversky Loss for hard lesion mining (ISBI 2019)
│   ├── generalized_dice.py   # Generalized Dice Loss with volume weighting (DLMIA 2017)
│   ├── lovasz_loss.py        # Lovász-Hinge Loss submodular surrogate (CVPR 2018)
│   ├── boundary_loss.py      # Signed Distance Boundary / Surface Loss (MIDL 2019)
│   ├── edge_loss.py          # Edge-Aware Laplacian Contour Loss (PraNet, Psi-Net)
│   ├── combo_loss.py         # Combo Loss balancing input/output imbalance (CMIG 2019)
│   ├── hausdorff_loss.py     # Differentiable Approximate Hausdorff Distance Loss (IEEE TMI 2019)
│   ├── weighted_focal_iou.py # Weighted Focal + Weighted IoU (ColonFormer BMVC 2021)
│   └── tversky_bce.py        # Combined Tversky + BCE Loss (UACANet ACM MM 2021)
├── utils/
│   ├── __init__.py
│   ├── dataloader.py         # PolypDataset & TestDataset with synchronized transforms
│   ├── augmentation.py       # Joint augmentations (HFlip, VFlip, Rotate 90, ColorJitter)
│   ├── setup_dataset.py      # HuggingFace auto-downloader & local archive extractor
│   ├── metrics.py            # Mathematically rigorous mDice, mIoU, wFb, Sm, Em, MAE
│   ├── trainer.py            # PraNet multi-scale trainer with gradient clipping
│   ├── evaluator.py          # Cross-dataset benchmark evaluator with LaTeX exporter
│   ├── visualizer.py         # 4-panel diagnostic grid, overlays, and bar charts
│   ├── logger.py             # CSV logging & loss/metric curve generator
│   └── checkpoint.py         # Checkpointer supporting best.pth & periodic snapshots
├── snapshots/                # Checkpoints directory (auto-created)
├── results/                  # Predicted masks, metrics.csv, and metrics.tex (auto-created)
├── logs/                     # Training CSV logs and loss curves (auto-created)
└── visualizations/           # Diagnostic grids and overlay figures (auto-created)
```

---

## 🧩 Adding Custom Models & Loss Functions

### Adding a Model
Create a new file in `models/your_model.py`:
```python
import torch.nn as nn
from models import register_model

@register_model('my_model')
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ...
    def forward(self, x):
        # Return either single tensor or tuple for deep supervision
        return out
```
Then train immediately with:
```bash
python train.py --model my_model
```

### Adding a Loss
Create a new file in `loss/your_loss.py`:
```python
import torch
from loss import register_loss

@register_loss('my_loss')
def my_loss_fn(pred, target):
    # Compute and return scalar loss
    return loss
```
Then train immediately with:
```bash
python train.py --model pranet --loss my_loss
```

---

## 📊 Benchmark Reference Results (PraNet Reproduction)

| Dataset | Images | mDice | mIoU | $F_\beta^w$ | $S_m$ | $E_m$ | MAE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Kvasir-SEG** | 100 | 0.898 | 0.840 | 0.885 | 0.915 | 0.948 | 0.030 |
| **CVC-ClinicDB** | 62 | 0.899 | 0.849 | 0.896 | 0.936 | 0.979 | 0.009 |
| **CVC-ColonDB** | 380 | 0.709 | 0.640 | 0.696 | 0.820 | 0.896 | 0.045 |
| **CVC-300** | 60 | 0.871 | 0.797 | 0.843 | 0.925 | 0.972 | 0.010 |
| **ETIS-LaribPolypDB** | 196 | 0.628 | 0.567 | 0.600 | 0.794 | 0.841 | 0.031 |
| **Overall Mean** | 798 | **0.801** | **0.739** | **0.784** | **0.878** | **0.927** | **0.025** |

---

## 📜 Citations

```bibtex
@inproceedings{fan2020pranet,
  title={PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
  author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  booktitle={MICCAI},
  pages={263--273},
  year={2020}
}

@article{gao2021res2net,
  title={Res2Net: A New Multi-Scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  volume={43},
  number={2},
  pages={652--662},
  year={2021}
}
```
