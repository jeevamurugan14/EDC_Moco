import os
import logging
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
from methods.edc1 import EDC
from datasets.dataset import AD_Dataset_WithVal
from datasets.data_utils import get_data_loader
from models.edc_ssl import MoCo_R50_R50
from configs.config_busi import DATASET_DIR

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

    save_name = f"{args.save_name}_seed{seed}"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)

    logger = get_logger(save_name, save_path, "INFO")
    logger.warning(f"[BUSI] Seed={seed}  GPU={args.gpu}")

    ds_train = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
        val_fraction=0.2, seed=seed, train=True,
    )
    train_dset = ds_train.get_train_dset()
    val_dset   = ds_train.get_val_dset()

    eval_dset = AD_Dataset_WithVal(
        name=args.dataset, data_dir=args.data_dir,
        img_size=args.img_size, train=False,
    ).get_eval_dset()

    tl = np.array(train_dset.targets)
    el = np.array(eval_dset.targets)
    print(f"Train normal : {Counter(tl)[0]}")
    print(f"Val normal   : {len(val_dset)}")
    print(f"Test normal  : {Counter(el)[0]}  Abnormal: {Counter(el)[1]}")

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

    model = MoCo_R50_R50(
        moco_weights_path=args.moco_weights_path,
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
        freeze_encoder=args.freeze_encoder,
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
    logger.info(f"Device: {device}")
    runner.model = runner.model.to(device)
    args.device  = device

    runner.set_data_loader(loader_dict)

    if args.resume:
        runner.load_model(args.load_path)

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
    print(f"  BUSI - Seed {seed} - Final Results")
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
    print(f"Threshold: {thresh:.4f} (95th-pct of val-normal scores)")

    return final


def main_worker(gpu, args):
    args.gpu = gpu
    seeds = args.seeds if (hasattr(args,"seeds") and args.seeds) else [args.seed]
    all_results = []
    for seed in seeds:
        result = run_single_seed(args, seed)
        all_results.append(result)

    if len(all_results) > 1:
        keys = ["eval/AUC","eval/AP","eval/acc","eval/f1",
                "eval/recall","eval/precision","eval/specificity"]
        print(f"\n{'='*60}")
        print(f"  BUSI - Multi-Seed Summary (seeds={seeds})")
        print(f"{'='*60}")
        for k in keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:30s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SSL-EDC BUSI Revised")

    parser.add_argument("--save_dir",          type=str, default="./saved_models")
    parser.add_argument("--save_name",         type=str, default="edc_ssl_busi_revised")
    parser.add_argument("--resume",            action="store_true", default=False)
    parser.add_argument("--load_path",         type=str, default=None)
    parser.add_argument("-o","--overwrite",    action="store_true", default=True)
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
    parser.add_argument("--dataset",           type=str, default="busi")
    parser.add_argument("--train_sampler",     type=str, default="RandomSampler")
    parser.add_argument("--img_size",          type=int, default=256)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--moco_weights_path", type=str, required=True)
    parser.add_argument("--freeze_encoder",    type=str2bool, default=False)
    parser.add_argument("--warmup_iters",      type=int, default=200)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--seeds",             type=int, nargs="+", default=None)
    parser.add_argument("--gpu",               type=str, default="0")
    parser.add_argument("--c",                 type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    seeds = args.seeds if (hasattr(args, 'seeds') and args.seeds) else [args.seed]
    for seed in seeds:
        args.seed = seed
        args.save_name = f'edc_ssl_busi_seed{seed}'
        save_path = os.path.join(args.save_dir, args.save_name)
        if os.path.exists(save_path) and args.overwrite and not args.resume:
            shutil.rmtree(save_path)
        print(f'\n=== Running BUSI seed {seed} ===')
        main_worker(int(args.gpu), args)

    if len(seeds) > 1:
        print(f'\nMulti-seed BUSI completed for seeds: {seeds}')
