"""
runners_edc_ssl/edc_ssl_busi_pixel_localization.py
B1: Pixel-level localization evaluation (Dice, IoU) + qualitative heatmaps.
Dataset: BUSI (combined benign+malignant test set, all masked lesions).

Since no checkpoint-saving step exists in the training loop, this script
trains the model exactly as your main BUSI run does, then -- instead of
discarding the per-pixel anomaly map after pooling it to an image-level
score -- keeps that map, upsamples it to the original image resolution,
thresholds it with Otsu's method (unsupervised, no pixel-level labels
needed), and compares it against the ground-truth lesion mask to compute
Dice and IoU. Also saves 5 qualitative heatmap examples: a true positive,
a false positive, a false negative, a subtle/hard case, and a normal case.
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from utils import get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc_revised import EDC
from datasets.dataset_with_val import AD_Dataset_WithVal
from datasets.data_utils import get_data_loader
from models.edc_ssl import MoCo_R50_R50
from configs.config_busi import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    if v.lower() in ("no", "false", "f", "n", "0"): return False
    raise ValueError("Boolean value expected.")


def dice_iou(pred_mask, gt_mask):
    """pred_mask, gt_mask: binary numpy arrays, same shape."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
    iou = intersection / (union + 1e-8)
    return dice, iou


def otsu_threshold_map(amap_2d):
    """amap_2d: float numpy array (H, W). Returns binary mask via Otsu."""
    norm = (amap_2d - amap_2d.min()) / (amap_2d.max() - amap_2d.min() + 1e-8)
    norm_u8 = (norm * 255).astype(np.uint8)
    _, binary = cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.uint8)


def save_heatmap_overlay(image_rgb, amap_2d, save_path):
    """image_rgb: (H, W, 3) uint8. amap_2d: float (H, W)."""
    norm = (amap_2d - amap_2d.min()) / (amap_2d.max() - amap_2d.min() + 1e-8)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * image_rgb + 0.5 * heatmap).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    save_name = f"busi_pixel_localization_seed{args.seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[BUSI Pixel-Localization] Seed={args.seed}  GPU={args.gpu}")

    ds_train = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
        val_fraction=0.2, seed=args.seed, train=True,
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
    generator.manual_seed(args.seed)

    loader_dict = {}
    loader_dict["train"] = get_data_loader(
        train_dset, args.batch_size, data_sampler=args.train_sampler,
        num_iters=args.num_train_iter, num_workers=args.num_workers,
        distributed=False, generator=generator,
    )
    loader_dict["val"] = get_data_loader(val_dset, args.eval_batch_size,
                                          num_workers=args.num_workers, drop_last=False)
    loader_dict["eval"] = get_data_loader(eval_dset, args.eval_batch_size,
                                           num_workers=args.num_workers, drop_last=False)

    model = MoCo_R50_R50(
        moco_weights_path=args.moco_weights_path,
        img_size=args.img_size, train_encoder=True, stop_grad=True,
        reshape=True, bn_pretrain=False, freeze_encoder=False,
        warmup_iters=args.warmup_iters,
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    runner = EDC(model=model, num_eval_iter=args.num_eval_iter, logger=logger)
    logger.info(f"Trainable params: {count_parameters(runner.model):,}")

    optimizer = get_optimizer_v2(
        runner.model, args.optim, args.lr, args.momentum,
        lr_encoder=args.lr_encoder, weight_decay=args.weight_decay,
    )
    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0,
    )
    runner.set_optimizer(optimizer, scheduler)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    runner.model = runner.model.to(device)
    args.device = device
    runner.set_data_loader(loader_dict)

    args.save_name = save_name
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    logger.info("Training proposed model (same protocol as main BUSI run)...")
    runner.train(args, device=device, logger=logger)

    final = runner.evaluate(args=args, device=device, val_loader=loader_dict["val"])
    logger.info(f"Image-level AUROC: {final['eval/AUC']:.4f}  AP: {final['eval/AP']:.4f}")

    # ---- Pixel-level pass: recompute anomaly maps, keep full spatial resolution ----
    logger.info("Running pixel-level localization pass...")
    runner.model.eval()

    mask_dir = args.mask_dir
    dice_scores, iou_scores = [], []
    qualitative_saved = {"TP": False, "FP": False, "FN": False, "subtle": False, "normal": False}
    qual_dir = os.path.join(save_path, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    # ---- DEBUG: show actual mask_dir contents and first few eval filenames ----
    mask_files_sample = sorted(os.listdir(mask_dir))[:5] if os.path.isdir(mask_dir) else []
    logger.info(f"[DEBUG] mask_dir={mask_dir}  sample files: {mask_files_sample}")
    debug_shown = 0

    with torch.no_grad():
        for idx, x, xo, y, filenames in loader_dict["eval"]:
            x = x.to(device)
            result = runner.model(x)
            amap = result["p_all"]  # (B, 1, h, w) or similar spatial map
            amap_up = F.interpolate(amap, size=(args.img_size, args.img_size), mode="bilinear")
            amap_up = amap_up.squeeze(1).cpu().numpy()  # (B, H, W)

            for i in range(x.shape[0]):
                fname = filenames[i]
                label = int(y[i].item())
                img_np = xo[i].numpy().astype("uint8")  # original image, (H, W, 3)
                amap_2d = amap_up[i]

                pred_mask = otsu_threshold_map(amap_2d)
                image_score = amap_2d.max()

                if label == 1:  # abnormal -> has a ground-truth mask
                    base = os.path.splitext(os.path.basename(fname))[0]
                    mask_path = os.path.join(mask_dir, base + "_mask.png")
                    if debug_shown < 5:
                        logger.info(f"[DEBUG] fname={fname!r}  base={base!r}  "
                                    f"mask_path={mask_path!r}  exists={os.path.exists(mask_path)}")
                        debug_shown += 1
                    if not os.path.exists(mask_path):
                        continue
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.resize(gt_mask, (args.img_size, args.img_size),
                                          interpolation=cv2.INTER_NEAREST)
                    gt_binary = (gt_mask > 127).astype(np.uint8)

                    dice, iou = dice_iou(pred_mask, gt_binary)
                    dice_scores.append(dice)
                    iou_scores.append(iou)

                    pred_positive = pred_mask.sum() > 0
                    if pred_positive and dice > 0.3 and not qualitative_saved["TP"]:
                        save_heatmap_overlay(img_np, amap_2d, os.path.join(qual_dir, "TP_example.png"))
                        qualitative_saved["TP"] = True
                    if dice < 0.05 and not qualitative_saved["FN"]:
                        save_heatmap_overlay(img_np, amap_2d, os.path.join(qual_dir, "FN_example.png"))
                        qualitative_saved["FN"] = True
                    if 0.05 <= dice < 0.3 and not qualitative_saved["subtle"]:
                        save_heatmap_overlay(img_np, amap_2d, os.path.join(qual_dir, "subtle_example.png"))
                        qualitative_saved["subtle"] = True
                else:  # normal image
                    if pred_mask.sum() > 0 and not qualitative_saved["FP"]:
                        save_heatmap_overlay(img_np, amap_2d, os.path.join(qual_dir, "FP_example.png"))
                        qualitative_saved["FP"] = True
                    if pred_mask.sum() == 0 and not qualitative_saved["normal"]:
                        save_heatmap_overlay(img_np, amap_2d, os.path.join(qual_dir, "normal_example.png"))
                        qualitative_saved["normal"] = True

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    std_dice = float(np.std(dice_scores)) if dice_scores else 0.0
    std_iou = float(np.std(iou_scores)) if iou_scores else 0.0

    print(f"\n{'='*60}")
    print("  BUSI Pixel-Level Localization Results")
    print(f"{'='*60}")
    print(f"  N lesions evaluated : {len(dice_scores)}")
    print(f"  Mean Dice           : {mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"  Mean IoU            : {mean_iou:.4f} +/- {std_iou:.4f}")
    print(f"  Qualitative examples saved to: {qual_dir}")
    for k, v in qualitative_saved.items():
        print(f"    {k:8s}: {'saved' if v else 'NOT FOUND (no matching case in this run)'}")

    os.makedirs("results/localization", exist_ok=True)
    result = {
        "Dataset": "BUSI", "Seed": args.seed,
        "N_lesions": len(dice_scores),
        "Mean_Dice": mean_dice, "Std_Dice": std_dice,
        "Mean_IoU": mean_iou, "Std_IoU": std_iou,
        "Image_AUROC": final["eval/AUC"], "Image_AP": final["eval/AP"],
    }
    pd.DataFrame([result]).to_csv(
        f"results/localization/busi_pixel_localization_seed{args.seed}.csv",
        index=False, float_format="%.4f"
    )
    print(f"Saved: results/localization/busi_pixel_localization_seed{args.seed}.csv")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BUSI Pixel-Level Localization (B1)")

    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num_train_iter", type=int, default=1000)
    parser.add_argument("--num_eval_iter", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lambda_diff", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--mask_dir", type=str,
                         default="/home/cs24d0008/EDC_SSL/BUSI/masks")
    parser.add_argument("--dataset", type=str, default="busi")
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--moco_weights_path", type=str, required=True)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--c", type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    run(args)