"""Leakage-safe experimental training of CMUNeXt and CMUNeXt-Plus.

Main improvement vs legacy train.py: validation comes from TrainDataset,
not from CVC-ClinicDB TestDataset. Test datasets remain unseen until a final run.
"""
import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from models import get_model
from loss import get_loss
from utils.setup_dataset import setup_dataset
from utils.dataloader import PolypDataset
from utils.metrics import MetricCalculator
from utils.logger import Logger
from utils.checkpoint import Checkpointer
from utils.evaluator import Evaluator


class BinaryMaskDataset(PolypDataset):
    """Override GT interpolation without modifying legacy pipeline results."""
    def __init__(self, image_root, gt_root, trainsize=352, is_train=True):
        super().__init__(image_root, gt_root, trainsize=trainsize, is_train=is_train)
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])


def make_split(root, batch_size, trainsize, seed, val_fraction, workers=2):
    images = os.path.join(root, 'TrainDataset', 'images')
    masks = os.path.join(root, 'TrainDataset', 'masks')
    train_ds = BinaryMaskDataset(images, masks, trainsize=trainsize, is_train=True)
    val_ds = BinaryMaskDataset(images, masks, trainsize=trainsize, is_train=False)
    assert train_ds.images == val_ds.images, 'Dataset indexing differs for train/val'
    indices = np.random.default_rng(seed).permutation(len(train_ds))
    n_val = max(1, int(round(len(indices) * val_fraction)))
    assert n_val < len(indices)
    train_idx, val_idx = indices[n_val:].tolist(), indices[:n_val].tolist()
    assert not set(train_idx) & set(val_idx)
    args = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True, drop_last=True, **args)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                            shuffle=False, drop_last=False, **args)
    print(f'[Split] TrainDataset={len(indices)}, train={len(train_idx)}, val={len(val_idx)}, seed={seed}')
    print('[Split] No TestDataset images used for checkpoint selection.')
    return train_loader, val_loader


def get_main(pred):
    return pred[-1] if isinstance(pred, (list, tuple)) else pred


def deep_supervision_loss(pred, gt, criterion):
    if not isinstance(pred, (list, tuple)):
        return criterion(pred, gt)
    # Deep-supervision heads are returned coarse -> fine -> final.
    if len(pred) == 4:
        weights = (0.15, 0.20, 0.30, 1.00)
    else:
        weights = [1 / len(pred)] * len(pred)
    return sum(float(w) * criterion(p, gt) for w, p in zip(weights, pred))


def train_epoch(model, loader, criterion, optimizer, scaler, device, trainsize, fast=False, amp=True):
    model.train()
    rates = (1.0,) if fast else (0.75, 1.0, 1.25)
    running = 0.0
    count = 0
    for images, masks in tqdm(loader, desc='train', leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        for rate in rates:
            size = int(round(trainsize * rate / 32) * 32)
            x = images if rate == 1 else F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False)
            y = masks if rate == 1 else F.interpolate(masks, size=(size, size), mode='nearest')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(x)
                loss = deep_supervision_loss(output, y, criterion)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if rate == 1.0:
                running += float(loss.item()) * images.shape[0]
                count += images.shape[0]
    return running / max(1, count)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    metrics = MetricCalculator()
    for images, masks in tqdm(loader, desc='validate', leave=False):
        output = get_main(model(images.to(device)))
        prediction = torch.sigmoid(output).float().cpu().numpy()[:, 0]
        gt = masks.numpy()[:, 0]
        # Same per-image min-max normalization and binary threshold as Evaluator.
        for p, g in zip(prediction, gt):
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            metrics.update(p, g)
    return metrics.get_results()


def run(cfg, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = setup_dataset(cfg.data_root)
    train_loader, val_loader = make_split(root, cfg.batchsize, cfg.trainsize,
                                         seed, cfg.val_fraction, cfg.workers)
    model = get_model(cfg.model).to(device)
    if cfg.init_weights:
        print(f'[Init] Initializing shared backbone/decoder from: {cfg.init_weights}')
        Checkpointer.load(model, cfg.init_weights, device=device)  # NO old optimizer state
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and not cfg.no_amp else None
    start_epoch = 1
    tag = cfg.model if len(cfg.seeds) == 1 else f'{cfg.model}_seed{seed}'
    if cfg.resume:
        resumed_epoch, _ = Checkpointer.load(model, cfg.resume, device=device, optimizer=optimizer)
        start_epoch = resumed_epoch + 1
        print(f'[Resume] Continuing {tag} from epoch {start_epoch}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                           eta_min=cfg.min_lr)
    # Reproduce LR phase if a SAME-MODEL checkpoint is resumed.
    for _ in range(start_epoch - 1):
        scheduler.step()
    logger = Logger(log_dir='./logs', model_name=tag)
    checkpointer = Checkpointer(save_dir='./snapshots', model_name=tag)
    if cfg.resume and os.path.exists(cfg.resume):
        # Prevent an untrained resumed step from replacing the previous best.
        import pathlib
        previous_best = pathlib.Path(checkpointer.get_best_path())
        if previous_best.exists():
            previous = torch.load(previous_best, map_location='cpu', weights_only=False)
            checkpointer.best_dice = float(previous.get('best_dice', previous.get('metrics', {}).get('mDice', -1.)))
    criterion = get_loss(cfg.loss)
    best = checkpointer.best_dice
    bad_epochs = 0
    print(f'[Run] model={cfg.model}, loss={cfg.loss}, device={device}, seed={seed}, epochs={cfg.epochs}')
    for epoch in range(start_epoch, cfg.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                 cfg.trainsize, fast=cfg.fast, amp=scaler is not None)
        valid = validate(model, val_loader, device)
        logger.log(epoch, {'loss': train_loss, 'lr': lr}, valid)
        checkpointer.save(model, optimizer, epoch, valid)
        if valid['mDice'] > best + cfg.min_delta:
            best = valid['mDice']
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step()
        if bad_epochs >= cfg.patience:
            print(f'[EarlyStop] no significant validation improvement in {cfg.patience} epochs')
            break
    logger.close()
    best_path = checkpointer.get_best_path()
    print(f'[Done] checkpoint={best_path} | best validation Dice={checkpointer.best_dice:.4f}')
    if cfg.eval_test:
        print('[FINAL TEST] Use only once after freezing experimental choices.')
        Checkpointer.load(model, best_path, device=device)
        evaluator = Evaluator(data_root=root, model=model, device=device,
                              testsize=cfg.trainsize, model_name=tag, weights_path=best_path)
        results = evaluator.evaluate_all(save_results_dir=os.path.join('./results', tag))
        evaluator.print_table(results, model_name=tag)
        evaluator.save_csv(results, path=os.path.join('./results', tag, 'metrics.csv'))


def main():
    parser = argparse.ArgumentParser(description='Experimental CMUNeXt+ training without test-data leakage')
    parser.add_argument('--model', default='cmunext_plus', choices=['cmunext', 'cmunext_plus'])
    parser.add_argument('--loss', default='structure_loss')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--init_weights', default=None, help='Warm-start from CMUNeXt base best.pth; epoch starts at 1')
    parser.add_argument('--resume', default=None, help='Resume SAME model and optimizer checkpoint')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--fast', action='store_true', help='One resolution per batch; fast smoke tests')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--eval_test', action='store_true', help='Evaluate the held-out five datasets AFTER final model selection')
    args = parser.parse_args()
    if args.init_weights and args.resume:
        parser.error('Choose only one of --init_weights or --resume')
    if not 0 < args.val_fraction < 0.5:
        parser.error('--val_fraction must be in (0, 0.5)')
    for seed in args.seeds:
        run(args, seed)


if __name__ == '__main__':
    main()