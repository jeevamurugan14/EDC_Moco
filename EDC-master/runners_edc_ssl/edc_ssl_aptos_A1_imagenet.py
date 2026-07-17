"""
runners_edc_ssl/edc_ssl_aptos_A1_imagenet.py
A1: ImageNet init instead of MoCo pretraining
Dataset: APTOS

FIX (protocol consistency with revised main results + 3-seed parity):
- Switched from `methods.edc1` (old, test-threshold / leaky) to
  `methods.edc_revised` (val-threshold, leakage-free), matching A5 and
  the main results table.
- Added aggregate_and_save() so 3-seed runs (--seeds 42 123 456) produce
  a mean/std summary AND a paper-ready row (all 6 metrics: AUROC, AP,
  Accuracy, Precision, Recall, F1) exactly like A5.
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, classification_report

from utils import get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc_revised import EDC   # FIX: leakage-free protocol, matches main results
from datasets.dataset_with_val import AD_Dataset_WithVal
from datasets.data_utils import get_data_loader
from models.edc import R50_R50
from configs.config_aptos import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes","true","t","y","1"): return True
    if v.lower() in ("no","false","f","n","0"): return False
    raise ValueError("Boolean value expected.")



def run_single_seed(args, seed):
    args.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    save_name = f"ablation_aptos_A1_imagenet_seed{seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[APTOS A1-ImageNet] Seed={seed}  GPU={args.gpu}")

    ds_train = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
        val_fraction=0.2, seed=seed, train=True,
    )
    train_dset = ds_train.get_train_dset()
    val_dset   = ds_train.get_val_dset()
    eval_dset  = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, train=False,
    ).get_eval_dset()

    tl = np.array(train_dset.targets)
    el = np.array(eval_dset.targets)
    print(f"Train: {Counter(tl)[0]} | Val: {len(val_dset)} | "
          f"Test Normal: {Counter(el)[0]} Abnormal: {Counter(el)[1]}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_dict = {}
    loader_dict["train"] = get_data_loader(
        train_dset, args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
        distributed=False, generator=generator,
    )
    loader_dict["val"] = get_data_loader(
        val_dset, args.eval_batch_size,
        num_workers=args.num_workers, drop_last=False,
    )
    loader_dict["eval"] = get_data_loader(
        eval_dset, args.eval_batch_size,
        num_workers=args.num_workers, drop_last=False,
    )

    # A1: ImageNet pretrained encoder (NO MoCo)
    model = R50_R50(
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
    )
    print("Model: ImageNet-pretrained R50_R50 (A1 Baseline)")

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    runner = EDC(
        model=model,
        num_eval_iter=args.num_eval_iter,
        logger=logger,
        amap_reduction="max",
    )
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
    args.device  = device
    runner.set_data_loader(loader_dict)

    args.save_name = save_name
    runner.tb_log  = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    runner.train(args, device=device, logger=logger)

    final = runner.evaluate(
        args=args, device=device,
        val_loader=loader_dict["val"],
    )

    y_true  = np.array(final["eval/y_true"])
    y_score = np.array(final["eval/y_score"])
    thresh  = final["eval/best_thr"]
    y_pred  = (y_score >= thresh).astype(int)

    print(f"\n{'='*60}")
    print(f"  APTOS | A1_ImageNet | Seed {seed}")
    print(f"{'='*60}")
    metrics_df = pd.DataFrame({
        "Metric": ["AUROC","Avg Precision","Accuracy",
                   "Precision","Recall","F1-score","Specificity"],
        "Value":  [final["eval/AUC"], final["eval/AP"], final["eval/acc"],
                   final["eval/precision"], final["eval/recall"],
                   final["eval/f1"], final["eval/specificity"]],
    })
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    print(f"\n{classification_report(y_true, y_pred, target_names=['NORMAL','ABNORMAL'], digits=4)}")

    os.makedirs("results/ablation", exist_ok=True)
    result = {
        "Variant": "A1_ImageNet", "Dataset": "APTOS", "Seed": seed,
        "AUROC": final["eval/AUC"], "AP": final["eval/AP"],
        "Accuracy": final["eval/acc"], "Precision": final["eval/precision"],
        "Recall": final["eval/recall"], "F1": final["eval/f1"],
        "Specificity": final["eval/specificity"],
    }
    pd.DataFrame([result]).to_csv(
        f"results/ablation/{save_name}.csv", index=False, float_format="%.4f"
    )
    print(f"Saved: results/ablation/{save_name}.csv")
    return final


def aggregate_and_save(all_results, seeds, variant, dataset):
    """
    Compute mean/std across seeds and save BOTH:
      1. A machine-readable CSV (mean + std columns) for downstream use.
      2. A 'paper-ready' single-row CSV matching tab:main_results / tab:ablation
         column format (mean only, 4 decimal places), so this row can be pasted
         directly into the LaTeX table with no manual re-typing/rounding.
    """
    key_map = {
        "AUROC":      "eval/AUC",
        "AP":         "eval/AP",
        "Accuracy":   "eval/acc",
        "Precision":  "eval/precision",
        "Recall":     "eval/recall",
        "F1":         "eval/f1",
        "Specificity":"eval/specificity",
    }

    means, stds = {}, {}
    for nice_name, k in key_map.items():
        vals = [r[k] for r in all_results]
        means[nice_name] = float(np.mean(vals))
        stds[nice_name]  = float(np.std(vals))

    print(f"\n{'='*60}")
    print(f"  {dataset} {variant} — 3-Seed Summary  seeds={seeds}")
    print(f"{'='*60}")
    for nice_name in key_map:
        print(f"  {nice_name:12s}: {means[nice_name]:.4f} +/- {stds[nice_name]:.4f}")

    os.makedirs("results/ablation", exist_ok=True)

    detail_rows = []
    for r, s in zip(all_results, seeds):
        row = {"Variant": variant, "Dataset": dataset, "Seed": s}
        for nice_name, k in key_map.items():
            row[nice_name] = r[k]
        detail_rows.append(row)
    detail_path = f"results/ablation/{dataset.lower()}_{variant}_3seed_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, float_format="%.4f")

    summary_row = {"Variant": variant, "Dataset": dataset, "Seeds": str(seeds)}
    for nice_name in key_map:
        summary_row[f"{nice_name}_mean"] = means[nice_name]
        summary_row[f"{nice_name}_std"]  = stds[nice_name]
    summary_path = f"results/ablation/{dataset.lower()}_{variant}_3seed_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, float_format="%.4f")

    paper_row = {
        "Dataset": dataset,
        "Variant": variant,
        "AUROC":     round(means["AUROC"], 4),
        "AP":        round(means["AP"], 4),
        "Accuracy":  round(means["Accuracy"], 4),
        "Precision": round(means["Precision"], 4),
        "Recall":    round(means["Recall"], 4),
        "F1-score":  round(means["F1"], 4),
    }
    paper_path = f"results/ablation/{dataset.lower()}_{variant}_paper_row.csv"
    pd.DataFrame([paper_row]).to_csv(paper_path, index=False, float_format="%.4f")

    print(f"\nSaved detail : {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved paper row (drop directly into tab:ablation): {paper_path}")

    return means, stds


def main_worker(args, variant, dataset):
    seeds = args.seeds if (hasattr(args,"seeds") and args.seeds) else [args.seed]
    all_results = []
    for seed in seeds:
        result = run_single_seed(args, seed)
        all_results.append(result)

    if len(all_results) > 1:
        aggregate_and_save(all_results, seeds, variant=variant, dataset=dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APTOS Ablation A1 ImageNet")

    parser.add_argument("--save_dir",          type=str, default="./saved_models")
    parser.add_argument("--use_tensorboard",   action="store_true", default=True)
    parser.add_argument("--epoch",             type=int,   default=1)
    parser.add_argument("--num_train_iter",    type=int,   default=1000)
    parser.add_argument("--num_eval_iter",     type=int,   default=250)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--eval_batch_size",   type=int,   default=64)
    parser.add_argument("--optim",             type=str,   default="AdamW")
    parser.add_argument("--lr",                type=float, default=5e-4)
    parser.add_argument("--lr_encoder",        type=float, default=1e-5)
    parser.add_argument("--momentum",          type=float, default=0.9)
    parser.add_argument("--weight_decay",      type=float, default=1e-4)
    parser.add_argument("--amp",               type=str2bool, default=False)
    parser.add_argument("--clip",              type=float, default=1.0)
    parser.add_argument("--data_dir",          type=str, default=DATASET_DIR)
    parser.add_argument("--dataset",           type=str, default="aptos")
    parser.add_argument("--train_sampler",     type=str, default="RandomSampler")
    parser.add_argument("--img_size",          type=int, default=256)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--warmup_iters",      type=int, default=0)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--seeds",             type=int, nargs="+", default=None)
    parser.add_argument("--gpu",               type=str, default="0")
    parser.add_argument("--resume",            action="store_true", default=False)
    parser.add_argument("--c",                 type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    main_worker(args, variant="A1_ImageNet", dataset="APTOS")