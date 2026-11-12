"""
runners_edc_ssl/edc_ssl_busi_baseline_patchcore.py
Baseline B2-4: PatchCore (Towards Total Recall in Industrial Anomaly Detection).
Dataset: BUSI

No gradient training. A frozen, ImageNet-pretrained ResNet-50 extracts
mid-level patch features (layer2 + layer3, upsampled and concatenated)
from normal training images. A memory bank of these patch features is
built and reduced via random-projection coreset subsampling (a
standard, efficient approximation to the greedy minimax-facility-
location coreset used in the original PatchCore paper). At test time,
each test-image patch is scored by its distance to its nearest
neighbour in the memory bank; the image-level anomaly score is the
maximum patch score (following the original paper's "core patch"
selection). Same data splits, same val-based threshold, same metrics
as the other baselines, for a fair same-protocol comparison.
To adapt to BUSI / APTOS: copy this file, change the DATASET_DIR
import and --dataset default, exactly as with the other baselines.
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

from utils import get_logger, over_write_args_from_file
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


class PatchFeatureExtractor(nn.Module):
    """Frozen ResNet-50: returns concatenated, spatially-aligned layer2+layer3 features."""
    def __init__(self):
        super().__init__()
        try:
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            backbone = tv_models.resnet50(pretrained=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2   # 512 ch
        self.layer3 = backbone.layer3   # 1024 ch
        self.avgpool_agg = nn.AvgPool2d(3, stride=1, padding=1)  # local patch aggregation

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f2 = self.avgpool_agg(f2)
        f3 = self.avgpool_agg(f3)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="nearest")
        patch_feat = torch.cat([f2, f3_up], dim=1)  # (B, 1536, H, W)
        return patch_feat


def run_single_seed(args, seed):
    args.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    save_name = f"baseline_patchcore_busi_seed{seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[BUSI Baseline-PatchCore] Seed={seed} GPU={args.gpu}")

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

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = get_data_loader(val_dset, args.eval_batch_size,
                                  num_workers=args.num_workers, drop_last=False)
    eval_loader = get_data_loader(eval_dset, args.eval_batch_size,
                                   num_workers=args.num_workers, drop_last=False)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    extractor = PatchFeatureExtractor().to(device).eval()

    # ---- Build memory bank from ALL normal training patches ----
    logger.info("Extracting patch features from normal training images...")
    all_patches = []
    with torch.no_grad():
        for _, x, _, _, _ in train_loader:
            x = x.to(device)
            feat = extractor(x)                      # (B, C, H, W)
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            all_patches.append(feat.cpu())
    memory_bank = torch.cat(all_patches, dim=0)       # (N_total_patches, C)
    logger.info(f"Raw memory bank size: {memory_bank.shape[0]} patches, dim={memory_bank.shape[1]}")

    # ---- Coreset subsampling (random-projection approximation) ----
    # NOTE: the original PatchCore paper uses greedy minimax-facility-location
    # coreset selection. Exact greedy selection scales poorly with memory-bank
    # size; we use random subsampling to a fixed coreset size as a standard,
    # efficient approximation used in several open re-implementations, and
    # document this simplification explicitly here.
    n_coreset = min(args.coreset_size, memory_bank.shape[0])
    torch.manual_seed(seed)
    idx = torch.randperm(memory_bank.shape[0])[:n_coreset]
    coreset = memory_bank[idx].to(device)             # (n_coreset, C)
    logger.info(f"Coreset size after subsampling: {coreset.shape[0]} patches")

    def score_batch(x):
        """Return (B, H, W) nearest-neighbour-distance anomaly maps for a batch."""
        with torch.no_grad():
            feat = extractor(x)                       # (B, C, H, W)
        Bx, Cx, Hx, Wx = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(Bx, Hx * Wx, Cx)  # (B, HW, C)
        dist_maps = []
        for b in range(Bx):
            # pairwise L2 distance: (HW, n_coreset)
            d = torch.cdist(flat[b], coreset, p=2.0)
            min_d, _ = d.min(dim=1)                    # nearest neighbour distance per patch
            dist_maps.append(min_d.reshape(Hx, Wx).cpu().numpy())
        return np.stack(dist_maps, axis=0)

    def reduce_map(amap, reduction="max"):
        flat = amap.reshape(amap.shape[0], -1)
        return flat.mean(1) if reduction == "mean" else flat.max(1)

    # ---- Threshold: 95th percentile of VALIDATION-NORMAL scores only ----
    logger.info("Scoring validation-normal images for threshold...")
    val_scores = []
    for _, x, _, _, _ in val_loader:
        x = x.to(device)
        amap = score_batch(x)
        val_scores.extend(reduce_map(amap, args.amap_reduction).tolist())
    val_scores = np.array(val_scores)
    thresh = float(np.percentile(val_scores, 95.0))
    print(f"[Threshold] From val-normal 95th-pct ({len(val_scores)} images): {thresh:.4f}")

    # ---- Test-set evaluation ----
    logger.info("Scoring test set...")
    y_true, y_prob = [], []
    for _, x, _, y, _ in eval_loader:
        x = x.to(device)
        amap = score_batch(x)
        score = reduce_map(amap, args.amap_reduction)
        y_true.extend(y.tolist())
        y_prob.extend(score.tolist())

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
    print(f"\n===== CONFUSION MATRIX (Baseline PatchCore, BUSI, seed={seed}) =====")
    print(f"TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"F1={f1:.4f}  Specificity={specificity:.4f}  AUROC={AUC:.4f}  AP={AP:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "ABNORMAL"], digits=4))

    os.makedirs("results/baselines", exist_ok=True)
    result = {
        "Baseline": "PatchCore", "Dataset": "BUSI", "Seed": seed,
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
    parser = argparse.ArgumentParser(description="BUSI Baseline: PatchCore")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--dataset", type=str, default="busi")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amap_reduction", type=str, default="max")
    parser.add_argument("--coreset_size", type=int, default=10000,
                         help="Number of patch features retained in the memory bank after subsampling")
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
        print(f"\n{'='*60}\n  BUSI Baseline-PatchCore — 3-Seed Summary\n{'='*60}")
        for k in keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()