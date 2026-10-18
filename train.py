"""
End-to-End Training Entry Point for Polyp Segmentation.

Usage:
    python train.py --model pranet --loss structure_loss --seed 42
    python train.py --model polyp_pvt --loss structure_loss --seed 42

Hyperparameter settings reproducing each paper exactly:
    Epochs: 100  (PraNet paper: 20; Polyp-PVT Table II: 100; we default 100)
    Initial Learning Rate: 1e-4
    Optimizer: Adam for PraNet/SANet/HarDNet-MSEG/UNet variants
               AdamW (weight_decay=1e-4) for Polyp-PVT  [Table II]
    Batch Size: 16
    Input Resolution: 352x352 (with multi-scale [0.75, 1.0, 1.25] per batch)
    Gradient Clipping: 0.5
    LR Decay: factor 0.1 every 50 epochs
    Deep Supervision: lateral_map_5, 4, 3, 2
"""

import argparse
import os
import sys
import random
import numpy as np
import torch

# Ensure the repository root is in sys.path regardless of execution CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model, list_models
from loss import get_loss, list_losses
from utils.setup_dataset import setup_dataset
from utils.dataloader import get_loader
from utils.trainer import Trainer
from utils.logger import Logger
from utils.checkpoint import Checkpointer
from utils.evaluator import Evaluator


def set_seed(seed: int = 42):
    """Ensure strict experimental reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train Polyp Segmentation Network")
    parser.add_argument('--model', type=str, default='pranet', help=f'Model name. Available: {list_models()}')
    parser.add_argument('--loss', type=str, default='structure_loss', help=f'Loss name. Available: {list_losses()}')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint (.pth) to resume training from. '
                             'Restores model weights, optimizer state, and starting epoch.')
    args = parser.parse_args()

    # 1. Enforce Fixed PraNet Hyperparameters
    args.lr = 1e-4
    args.batchsize = 16
    args.trainsize = 352
    args.clip = 0.5
    args.decay_rate = 0.1
    args.decay_epoch = 50

    # 2. Reproducibility
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"POLYP SEGMENTATION TRAINING PIPELINE")
    print(f"============================================================")
    print(f"Model: {args.model} | Loss: {args.loss} | Seed: {args.seed} | Device: {device}")
    print(f"Resolution: {args.trainsize}x{args.trainsize} (Multi-scale: 0.75, 1.0, 1.25)")
    print(f"LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batchsize}")
    print(f"============================================================")


    # 3. Setup Dataset
    data_root = setup_dataset(args.data_root)
    train_img_dir = os.path.join(data_root, 'TrainDataset', 'images')
    train_gt_dir = os.path.join(data_root, 'TrainDataset', 'masks')
    val_img_dir = os.path.join(data_root, 'TestDataset', 'CVC-ClinicDB', 'images')
    val_gt_dir = os.path.join(data_root, 'TestDataset', 'CVC-ClinicDB', 'masks')

    # 4. Data Loaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = get_loader(
        train_img_dir, train_gt_dir,
        batchsize=args.batchsize,
        trainsize=args.trainsize,
        shuffle=True,
        num_workers=num_workers,
        is_train=True
    )
    val_loader = get_loader(
        val_img_dir, val_gt_dir,
        batchsize=args.batchsize,
        trainsize=args.trainsize,
        shuffle=False,
        num_workers=num_workers,
        is_train=False
    )
    print(f"[Data] Training pairs: {len(train_loader.dataset)} | Validation pairs: {len(val_loader.dataset)}")

    # 5. Build Model & Loss
    print(f"[Init] Initializing model '{args.model}'...")
    model = get_model(args.model)
    loss_fn = get_loss(args.loss)

    # Per-paper optimizer dispatch:
    #   Polyp-PVT Table II: AdamW, weight_decay=1e-4
    #   PraNet / SANet / HarDNet-MSEG / UNet variants: Adam (no weight decay)
    _ADAMW_MODELS = {'polyp_pvt'}
    if args.model.lower() in _ADAMW_MODELS:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        opt_name = f"AdamW (weight_decay=1e-4)"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt_name = "Adam"


    # 6. Logger, Checkpointer, and Evaluator
    logger = Logger(log_dir='./logs', model_name=args.model)
    checkpointer = Checkpointer(save_dir='./snapshots', model_name=args.model)
    val_evaluator = Evaluator(data_root=data_root, model=model, device=device, testsize=args.trainsize)
    trainer = Trainer(model, optimizer, loss_fn, args, logger, checkpointer)

    # 7. Resume from checkpoint (optional)
    start_epoch = 1
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[Resume] WARNING: checkpoint '{args.resume}' not found. Starting from epoch 1.")
        else:
            resumed_epoch, resumed_metrics = Checkpointer.load(
                model, args.resume, device=device, optimizer=optimizer
            )
            start_epoch = resumed_epoch + 1
            checkpointer.best_dice = resumed_metrics.get('mDice', 0.0)
            print(f"[Resume] Resumed from epoch {resumed_epoch} "
                  f"(best mDice so far: {checkpointer.best_dice:.4f}). "
                  f"Continuing from epoch {start_epoch}.")

    # 8. Training Loop
    print(f"[Init] Optimizer: {opt_name} | lr={args.lr}")
    print(f"\n[Train] Starting training from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):

        train_stats = trainer.train_one_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, val_evaluator)
        logger.log(epoch, train_stats, val_metrics)
        checkpointer.save(model, optimizer, epoch, val_metrics)

    logger.close()

    # 8. Final Cross-Dataset Evaluation on Best Checkpoint
    best_weights = checkpointer.get_best_path()
    if os.path.exists(best_weights):
        print(f"\n============================================================")
        print(f"RUNNING FINAL BENCHMARK & EDGE AI EVALUATION ON BEST MODEL: {best_weights}")
        print(f"============================================================")
        Checkpointer.load(model, best_weights, device=device)
        evaluator = Evaluator(
            data_root=data_root,
            model=model,
            device=device,
            testsize=args.trainsize,
            model_name=args.model,
            weights_path=best_weights
        )
        results_dir = os.path.join('./results', args.model)

        # 1. Edge AI Deployment Profiling (Params, FLOPs, Latency, FPS, Peak Memory)
        print(f"\n[Evaluation] Profiling Edge AI & Hardware efficiency...")
        efficiency = evaluator.benchmark_efficiency(measure_cpu=True)

        # 2. Benchmark accuracy across all 5 test sets
        results = evaluator.evaluate_all(save_results_dir=results_dir)

        # 3. Print & save both Accuracy and Hardware metrics
        evaluator.print_table(results, efficiency=efficiency, model_name=args.model)
        evaluator.save_csv(results, path=os.path.join(results_dir, 'metrics.csv'), efficiency=efficiency)
        evaluator.save_latex(results, path=os.path.join(results_dir, 'metrics.tex'), efficiency=efficiency)

    print("\n[Done] Training and benchmark evaluation complete.")


if __name__ == '__main__':
    main()
