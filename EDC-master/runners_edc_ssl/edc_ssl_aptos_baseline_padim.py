"""
runners_edc_ssl/edc_ssl_lungct_baseline_padim.py
Baseline B2-2: PaDiM (Patch Distribution Modeling).
Dataset: APTOS

No gradient training. A frozen, ImageNet-pretrained ResNet-50 extracts
features from normal training images; a multivariate Gaussian is fit
per spatial location (mean + covariance, with a small random-projection
dimensionality reduction, exactly as in the original PaDiM paper). At
test time, Mahalanobis distance at each spatial location forms the
anomaly map. Same data splits, same val-based threshold, same metrics
as your main pipeline and the autoencoder baseline, for a fair
same-protocol comparison.

To adapt to BUSI / APTOS: copy this file, change the DATASET_DIR import
and --dataset default, exactly as with the autoencoder baseline scripts.
"""

import os
import random
from collections import Counter, OrderedDict

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
from scipy.spatial.distance import mahalanobis

from utils import get_logger, over_write_args_from_file
from datasets.dataset_with_val import AD_Dataset_WithVal
from datasets.data_utils import get_data_loader
from configs.config_aptos import DATASET_DIR

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
# Frozen ResNet-50 feature extractor: pulls layer1/layer2/layer3
# ------------------------------------------------------------------
class ResNetFeatureExtractor(nn.Module):
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
        self.layer1 = backbone.layer1   # 256 ch,  64x64  (for 256 input)
        self.layer2 = backbone.layer2   # 512 ch,  32x32
        self.layer3 = backbone.layer3   # 1024 ch, 16x16

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


def embed_concat(f1, f2, f3):
    """Upsample f2, f3 to f1's spatial size and concatenate channels."""
    _, _, H, W = f1.shape
    f2u = F.interpolate(f2, size=(H, W), mode="nearest")
    f3u = F.interpolate(f3, size=(H, W), mode="nearest")
    return torch.cat([f1, f2u, f3u], dim=1)   # (B, 256+512+1024, H, W)


def run_single_seed(args, seed):
    args.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    save_name = f"baseline_padim_aptos_seed{seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[APTOS Baseline-PaDiM] Seed={seed} GPU={args.gpu}")

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

    # No shuffling needed -- we just pass every normal training image through once
    # NOTE: PaDiM needs one deterministic pass over every normal training
    # image (no repeated sampling needed), so we bypass the custom
    # get_data_loader here and use a plain torch DataLoader directly --
    # your data_utils.get_data_loader only implements RandomSampler.
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = get_data_loader(val_dset, args.eval_batch_size,
                                  num_workers=args.num_workers, drop_last=False)
    eval_loader = get_data_loader(eval_dset, args.eval_batch_size,
                                   num_workers=args.num_workers, drop_last=False)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    extractor = ResNetFeatureExtractor().to(device).eval()

    # ---- Random projection: reduce concatenated channel dim (1792) to args.n_dims ----
    total_c = 256 + 512 + 1024
    torch.manual_seed(seed)  # keep projection reproducible per seed
    rand_idx = torch.randperm(total_c)[:args.n_dims]

    # ---- Pass 1: collect embeddings over ALL normal training images ----
    logger.info("Collecting normal-image embeddings for Gaussian fitting...")
    embedding_list = []
    with torch.no_grad():
        for _, x, _, _, _ in train_loader:
            x = x.to(device)
            f1, f2, f3 = extractor(x)
            emb = embed_concat(f1, f2, f3)          # (B, 1792, H, W)
            emb = emb[:, rand_idx, :, :]             # (B, n_dims, H, W)
            embedding_list.append(emb.cpu())
    embeddings = torch.cat(embedding_list, dim=0)    # (N, n_dims, H, W)
    B, C, H, W = embeddings.shape
    embeddings = embeddings.view(B, C, H * W).numpy()

    logger.info(f"Fitting per-location Gaussian over {B} normal images, "
                f"{H*W} locations, {C}-dim features...")
    mean = np.mean(embeddings, axis=0)               # (C, H*W)
    cov = np.zeros((C, C, H * W))
    identity = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embeddings[:, :, i], rowvar=False) + 0.01 * identity
    cov_inv = np.zeros_like(cov)
    for i in range(H * W):
        cov_inv[:, :, i] = np.linalg.inv(cov[:, :, i])

    def score_batch(x):
        """Return (B, H, W) Mahalanobis-distance anomaly maps for a batch."""
        with torch.no_grad():
            f1, f2, f3 = extractor(x)
            emb = embed_concat(f1, f2, f3)[:, rand_idx, :, :]
        emb = emb.cpu().numpy()
        Bx, Cx, Hx, Wx = emb.shape
        emb = emb.reshape(Bx, Cx, Hx * Wx)
        dist_list = []
        for b in range(Bx):
            dists = np.zeros(Hx * Wx)
            for i in range(Hx * Wx):
                delta = emb[b, :, i] - mean[:, i]
                dists[i] = mahalanobis(delta, np.zeros(Cx), cov_inv[:, :, i])
            dist_list.append(dists.reshape(Hx, Wx))
        return np.stack(dist_list, axis=0)  # (B, H, W)

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
    print(f"\n===== CONFUSION MATRIX (Baseline PaDiM, LungCT, seed={seed}) =====")
    print(f"TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"F1={f1:.4f}  Specificity={specificity:.4f}  AUROC={AUC:.4f}  AP={AP:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "ABNORMAL"], digits=4))

    os.makedirs("results/baselines", exist_ok=True)
    result = {
        "Baseline": "PaDiM", "Dataset": "APTOS", "Seed": seed,
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
    parser = argparse.ArgumentParser(description="APTOS Baseline: PaDiM")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--dataset", type=str, default="aptos")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amap_reduction", type=str, default="max")
    parser.add_argument("--n_dims", type=int, default=100,
                         help="Random-projection dimensionality (standard PaDiM default: 100)")
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
        print(f"\n{'='*60}\n  APTOS Baseline-PaDiM — 3-Seed Summary\n{'='*60}")
        for k in keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()