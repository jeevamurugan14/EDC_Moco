"""
runners_baselines/baseline_autoencoder_lungct.py
Baseline B2-1: Plain pixel-level convolutional autoencoder.
Dataset: BUSI

This is the SIMPLEST possible one-class anomaly detector: train an
encoder-decoder to reconstruct normal images at the PIXEL level (no MoCo,
no multi-scale feature reconstruction, no cosine-distance-in-feature-space).
Anomaly score = pixel-wise reconstruction error. Everything else --
data splits, val-based threshold, metrics -- matches your main pipeline
exactly, so this is a fair, same-protocol comparison.

To adapt to BUSI / APTOS: copy this file, change the DATASET_DIR import
and --dataset default, exactly like your existing A1-A5 scripts do.
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

from utils import get_logger, count_parameters, over_write_args_from_file
from datasets.dataset_with_val import AD_Dataset_WithVal
from datasets.data_utils import get_data_loader
from configs.config_busi import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    if v.lower() in ("no", "false", "f", "n", "0"): return False
    raise ValueError("Boolean value expected.")


# ------------------------------------------------------------------
# Model: plain convolutional autoencoder, operates on raw 256x256 RGB
# ------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 128
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),# 16
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),# 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), # 16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),    # 128
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),                                  # 256
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def pixel_anomaly_map(x, recon):
    """Per-pixel squared error, averaged over channels -> (B, H, W)."""
    return ((x - recon) ** 2).mean(dim=1)


def reduce_map(anomaly_map, reduction="max"):
    """anomaly_map: (B, H, W) -> (B,) image-level score."""
    flat = anomaly_map.flatten(1)
    if reduction == "mean":
        return flat.mean(1)
    return flat.max(1)[0]


def specificity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).astype(int)
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    N = (y_true == 0).sum()
    return TN / (N + 1e-8)


def run_single_seed(args, seed):
    args.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    save_name = f"baseline_ae_busi_seed{seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[BUSI Baseline-Autoencoder] Seed={seed} GPU={args.gpu}")

    # --- data: identical split protocol to your main pipeline ---
    ds_train = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
        val_fraction=0.2, seed=seed, train=True,
    )
    train_dset = ds_train.get_train_dset()
    val_dset = ds_train.get_val_dset()
    eval_dset = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, train=False,
    ).get_eval_dset()

    tl = np.array(train_dset.targets)
    el = np.array(eval_dset.targets)
    print(f"Train normal: {Counter(tl)[0]} | Val: {len(val_dset)} | "
          f"Test Normal: {Counter(el)[0]} Abnormal: {Counter(el)[1]}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = get_data_loader(
        train_dset, args.batch_size, data_sampler=args.train_sampler,
        num_iters=args.num_train_iter, num_workers=args.num_workers,
        distributed=False, generator=generator,
    )
    val_loader = get_data_loader(val_dset, args.eval_batch_size,
                                  num_workers=args.num_workers, drop_last=False)
    eval_loader = get_data_loader(eval_dset, args.eval_batch_size,
                                   num_workers=args.num_workers, drop_last=False)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    logger.info(f"Trainable params: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- train: pixel MSE reconstruction on normal images only ---
    model.train()
    it = 0
    data_iter = iter(train_loader)
    while it < args.num_train_iter:
        try:
            _, x, _, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            _, x, _, _, _ = next(data_iter)
        x = x.to(device)
        recon = model(x)
        loss = F.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (it + 1) % 100 == 0:
            logger.info(f"iter {it+1}/{args.num_train_iter}  recon_loss={loss.item():.6f}")
        it += 1

    # --- threshold: 95th percentile of VALIDATION-NORMAL reconstruction error only ---
    model.eval()
    with torch.no_grad():
        val_scores = []
        for _, x, _, _, _ in val_loader:
            x = x.to(device)
            recon = model(x)
            amap = pixel_anomaly_map(x, recon)
            val_scores.extend(reduce_map(amap, args.amap_reduction).cpu().tolist())
        val_scores = np.array(val_scores)
        thresh = float(np.percentile(val_scores, 95.0))
        print(f"[Threshold] From val-normal 95th-pct ({len(val_scores)} images): {thresh:.6f}")

        # --- test-set evaluation ---
        y_true, y_prob = [], []
        for _, x, _, y, _ in eval_loader:
            x = x.to(device)
            recon = model(x)
            amap = pixel_anomaly_map(x, recon)
            score = reduce_map(amap, args.amap_reduction)
            y_true.extend(y.tolist())
            y_prob.extend(score.cpu().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= thresh).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = specificity_score(y_true, y_pred)
    precision = float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-8))
    AUC = roc_auc_score(y_true, y_prob)
    AP = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n===== CONFUSION MATRIX (Baseline Autoencoder, BUSI, seed={seed}) =====")
    print(f"TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"F1={f1:.4f}  Specificity={specificity:.4f}  AUROC={AUC:.4f}  AP={AP:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "ABNORMAL"], digits=4))

    os.makedirs("results/baselines", exist_ok=True)
    result = {
        "Baseline": "Autoencoder", "Dataset": "BUSI", "Seed": seed,
        "AUROC": AUC, "AP": AP, "Accuracy": acc,
        "Precision": precision, "Recall": recall, "F1": f1, "Specificity": specificity,
    }
    pd.DataFrame([result]).to_csv(
        f"results/baselines/{save_name}.csv", index=False, float_format="%.4f"
    )
    print(f"Saved: results/baselines/{save_name}.csv")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BUSI Baseline: Plain Autoencoder")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--num_train_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--dataset", type=str, default="busi")
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amap_reduction", type=str, default="max")  # match your main pipeline's default
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--c", type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    seeds = args.seeds if args.seeds else [args.seed]
    all_results = []
    for seed in seeds:
        all_results.append(run_single_seed(args, seed))

    if len(all_results) > 1:
        keys = ["AUROC", "AP", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
        print(f"\n{'='*60}\n  BUSI Baseline-Autoencoder — 3-Seed Summary\n{'='*60}")
        for k in keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()