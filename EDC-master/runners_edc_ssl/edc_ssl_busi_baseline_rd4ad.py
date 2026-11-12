"""
runners_edc_ssl/edc_ssl_lungct_baseline_rd4ad.py
Baseline B2-3: RD4AD (Reverse Distillation for Anomaly Detection).
Dataset: BUSI

Structurally the closest baseline to your own EDC method -- which is
exactly why it matters for positioning novelty (Reviewer 3, point 1).
Key difference from your proposed method: the "teacher" encoder here
is FROZEN at ImageNet weights the entire time (no MoCo pretraining, no
encoder fine-tuning, no multi-scale contrastive supervision). Only a
lightweight bottleneck (OCBE) + mirrored student decoder are trained,
on normal images only, to reconstruct the frozen teacher's multi-scale
features. Anomaly score = cosine discrepancy between teacher and
student features, matching your own scoring convention. Same data
splits, same val-based threshold, same metrics as your other baselines.

To adapt to BUSI / APTOS: copy this file, change the DATASET_DIR import
and --dataset default, exactly as with the other baseline scripts.
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
import torchvision.models as tv_models
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


def specificity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).astype(int)
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    N = (y_true == 0).sum()
    return TN / (N + 1e-8)


# ------------------------------------------------------------------
# Frozen teacher: ImageNet ResNet-50, never updated
# ------------------------------------------------------------------
class FrozenTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            # older torchvision (pre-0.13) uses the legacy pretrained=True API
            backbone = tv_models.resnet50(pretrained=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch
        self.layer2 = backbone.layer2   # 512 ch
        self.layer3 = backbone.layer3   # 1024 ch

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        return e1, e2, e3


def conv_bn_relu(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


# ------------------------------------------------------------------
# OCBE: One-Class Bottleneck Embedding.
# Fuses teacher's 3 multi-scale features into a single compact code.
# ------------------------------------------------------------------
class OCBE(nn.Module):
    def __init__(self):
        super().__init__()
        # downsample e1, e2 to e3's spatial size, fuse channel-wise
        self.down1 = conv_bn_relu(256, 256, stride=2)   # e1: 64x64 -> 32x32
        self.down1b = conv_bn_relu(256, 256, stride=2)  #     32x32 -> 16x16
        self.down2 = conv_bn_relu(512, 512, stride=2)   # e2: 32x32 -> 16x16
        self.fuse = conv_bn_relu(256 + 512 + 1024, 2048, stride=2)  # -> 8x8, bottleneck

    def forward(self, e1, e2, e3):
        d1 = self.down1b(self.down1(e1))
        d2 = self.down2(e2)
        cat = torch.cat([d1, d2, e3], dim=1)
        z = self.fuse(cat)  # (B, 2048, 8, 8)
        return z


# ------------------------------------------------------------------
# Student decoder: mirrors the teacher, reconstructs e1, e2, e3 from z
# ------------------------------------------------------------------
class StudentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
        )  # -> 16x16, reconstructs e3
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )  # -> 32x32, reconstructs e2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )  # -> 64x64, reconstructs e1

    def forward(self, z):
        d3 = self.up3(z)
        d2 = self.up2(d3)
        d1 = self.up1(d2)
        return d1, d2, d3   # matches e1, e2, e3 spatial/channel shapes


def cosine_map(a, b):
    """Per-pixel (1 - cosine similarity) along channel dim -> (B, H, W)."""
    a_n = F.normalize(a, dim=1)
    b_n = F.normalize(b, dim=1)
    return 1 - (a_n * b_n).sum(dim=1)


def reduce_map(amap, reduction="max"):
    flat = amap.flatten(1)
    return flat.mean(1) if reduction == "mean" else flat.max(1)[0]


def run_single_seed(args, seed):
    args.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    save_name = f"baseline_rd4ad_busi_seed{seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[BUSI Baseline-RD4AD] Seed={seed} GPU={args.gpu}")

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
    teacher = FrozenTeacher().to(device).eval()
    ocbe = OCBE().to(device)
    student = StudentDecoder().to(device)

    trainable = list(ocbe.parameters()) + list(student.parameters())
    logger.info(f"Trainable params (OCBE+student, teacher frozen): {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Train: OCBE + student only, teacher stays frozen throughout ----
    ocbe.train(); student.train()
    it = 0
    data_iter = iter(train_loader)
    while it < args.num_train_iter:
        try:
            _, x, _, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            _, x, _, _, _ = next(data_iter)
        x = x.to(device)

        with torch.no_grad():
            e1, e2, e3 = teacher(x)   # frozen, no gradient

        z = ocbe(e1, e2, e3)
        d1, d2, d3 = student(z)

        loss = (
            cosine_map(d1, e1).mean() +
            cosine_map(d2, e2).mean() +
            cosine_map(d3, e3).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % 100 == 0:
            logger.info(f"iter {it+1}/{args.num_train_iter}  recon_loss={loss.item():.6f}")
        it += 1

    # ---- Scoring function: fused, upsampled cosine-discrepancy map ----
    def score_batch(x):
        with torch.no_grad():
            e1, e2, e3 = teacher(x)
            z = ocbe(e1, e2, e3)
            d1, d2, d3 = student(z)
            m1 = cosine_map(d1, e1)                                   # 64x64
            m2 = F.interpolate(cosine_map(d2, e2).unsqueeze(1),
                                size=m1.shape[-2:], mode="bilinear").squeeze(1)
            m3 = F.interpolate(cosine_map(d3, e3).unsqueeze(1),
                                size=m1.shape[-2:], mode="bilinear").squeeze(1)
            return (m1 + m2 + m3) / 3.0

    ocbe.eval(); student.eval()

    # ---- Threshold: 95th percentile of VALIDATION-NORMAL scores only ----
    val_scores = []
    for _, x, _, _, _ in val_loader:
        x = x.to(device)
        amap = score_batch(x)
        val_scores.extend(reduce_map(amap, args.amap_reduction).cpu().tolist())
    val_scores = np.array(val_scores)
    thresh = float(np.percentile(val_scores, 95.0))
    print(f"[Threshold] From val-normal 95th-pct ({len(val_scores)} images): {thresh:.4f}")

    # ---- Test-set evaluation ----
    y_true, y_prob = [], []
    for _, x, _, y, _ in eval_loader:
        x = x.to(device)
        amap = score_batch(x)
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
    print(f"\n===== CONFUSION MATRIX (Baseline RD4AD, LungCT, seed={seed}) =====")
    print(f"TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"F1={f1:.4f}  Specificity={specificity:.4f}  AUROC={AUC:.4f}  AP={AP:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "ABNORMAL"], digits=4))

    os.makedirs("results/baselines", exist_ok=True)
    result = {
        "Baseline": "RD4AD", "Dataset": "BUSI", "Seed": seed,
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
    parser = argparse.ArgumentParser(description="BUSI Baseline: RD4AD")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--num_train_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--dataset", type=str, default="busi")
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amap_reduction", type=str, default="max")
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
        print(f"\n{'='*60}\n  BUSI Baseline-RD4AD — 3-Seed Summary\n{'='*60}")
        for k in keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()