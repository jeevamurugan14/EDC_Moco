#!/usr/bin/env python3
"""
run_stage2.py -- GPU step of the CVIP revision.

For each seed it:
  1. looks for a saved Stage-2 checkpoint;
  2. if found  -> loads it and only runs inference (minutes);
     if missing -> retrains Stage 2 from the MoCo (Stage-1) checkpoint with
                   the settings of Section 4.2, then saves the checkpoint;
  3. writes  scores/<dataset>_seed<seed>.csv  (+ _levels.npz)
     with ONE model and ONE validation threshold for every test image,
     ready for eval_overall.py.

STEP 0 (no GPU, 1 second): see which checkpoints exist
    python run_stage2.py --dataset busi --data_root /data/Dataset_BUSI_with_GT \
        --ckpt_dir checkpoints --check_only

STEP 1: run (loads or retrains automatically)
    python run_stage2.py --dataset busi --data_root /data/Dataset_BUSI_with_GT \
        --moco_ckpt checkpoints/moco_joint.pth --ckpt_dir checkpoints
    python run_stage2.py --dataset lungct --data_root /data/chest_ct/Data \
        --moco_ckpt checkpoints/moco_joint.pth --ckpt_dir checkpoints

STEP 2 (CPU): python eval_overall.py --dataset busi --std ; same for lungct

IMPORTANT
* If your own training code/model class exists, prefer it for loading your
  checkpoints (numbers then match the paper exactly).  This file contains a
  reference implementation of the model described in the paper; if your
  checkpoint keys do not match it, the script stops and tells you.
* If you retrain, ALL tables (2-7) must use the new numbers, not only the
  new "Overall" row.
* Check the printed image counts against Table 1 before trusting results.

Expected data layout
  BUSI   : <root>/normal/*.png  <root>/benign/*.png  <root>/malignant/*.png
           (files containing "_mask" are ignored)
  LungCT : <root>/train/normal/*  (normal training pool)
           <root>/test/<class>/*  (class "normal" = normal, every other class = abnormal)
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_scores import dump  # noqa: E402  (same folder)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# =========================================================================
# Reproducibility
# =========================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# Data
# =========================================================================
def list_images(folder):
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*")))
             if f.lower().endswith(IMG_EXT) and "_mask" not in os.path.basename(f)]
    if not files:
        sys.exit(f"ERROR: no images found in {folder}")
    return files


def build_splits(args, seed):
    """Returns lists of (path, label) for train / val / test."""
    rng_fixed = random.Random(args.test_split_seed)   # test split: identical for every seed
    rng_seed = random.Random(seed)                      # val split: changes with the seed

    if args.dataset == "busi":
        normals = list_images(os.path.join(args.data_root, "normal"))
        rng_fixed.shuffle(normals)
        n_test = round(args.test_frac * len(normals))
        test_n, pool = normals[:n_test], normals[n_test:]
        test = [(p, "normal") for p in test_n]
        for c in ("benign", "malignant"):
            test += [(p, c) for p in list_images(os.path.join(args.data_root, c))]
    else:  # lungct
        pool = list_images(os.path.join(args.data_root, "train", "normal"))
        test = []
        for d in sorted(glob.glob(os.path.join(args.data_root, "test", "*"))):
            if os.path.isdir(d):
                lab = "normal" if os.path.basename(d).lower().startswith("normal") else "abnormal"
                test += [(p, lab) for p in list_images(d)]

    pool = sorted(pool); rng_seed.shuffle(pool)
    n_val = round(args.val_frac * len(pool))
    val = [(p, "normal") for p in pool[:n_val]]
    train = [(p, "normal") for p in pool[n_val:]]

    counts = {lab: sum(1 for _, l in test if l == lab) for lab in sorted({l for _, l in test})}
    print(f"[seed {seed}] train={len(train)}  val={len(val)}  test={counts}")
    expected = {"busi": (85, 21, 27), "lungct": (119, 29, 60)}[args.dataset]
    got = (len(train), len(val), counts.get("normal", 0))
    if got != expected:
        print(f"  WARNING: Table 1 says train/val/test-normal = {expected}, got {got}. "
              f"Use your ORIGINAL split (--split_file) if you have it.")
    return train, val, test


def load_split_file(path, seed):
    """Optional: a text file per seed with lines  'split,label,path'  (your original split)."""
    f = path.format(seed=seed)
    out = {"train": [], "val": [], "test": []}
    for line in open(f):
        line = line.strip()
        if line and not line.startswith("#"):
            sp, lab, p = line.split(",", 2)
            out[sp].append((p, lab))
    print(f"[seed {seed}] split file {f}: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    return out["train"], out["val"], out["test"]


TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageList(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        x = TF(Image.open(p).convert("RGB"))
        return x, lab, os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)


# =========================================================================
# Model (reference implementation of Section 3.2)
# =========================================================================
class Encoder(nn.Module):
    """ResNet50: e1=layer1 (256,64,64) e2=layer2 (512,32,32) e3=layer3 (1024,16,16) e4=layer4 (2048,8,8)."""
    def __init__(self):
        super().__init__()
        r = resnet50(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        x = self.stem(x)
        e1 = self.layer1(x); e2 = self.layer2(e1); e3 = self.layer3(e2); e4 = self.layer4(e3)
        return [e1, e2, e3, e4]


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))
        self.head = nn.Conv2d(cout, cout, 1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.head(self.body(x))


class Decoder(nn.Module):
    """e4 -> d3 (1024,16) -> d2 (512,32) -> d1 (256,64)."""
    def __init__(self, use_skip):
        super().__init__()
        self.use_skip = use_skip
        s = (lambda c: c) if use_skip else (lambda c: 0)
        self.up3 = UpBlock(2048, s(1024), 1024)
        self.up2 = UpBlock(1024, s(512), 512)
        self.up1 = UpBlock(512, s(256), 256)

    def forward(self, enc):
        e1, e2, e3, e4 = enc
        d3 = self.up3(e4, e3 if self.use_skip else None)
        d2 = self.up2(d3, e2 if self.use_skip else None)
        d1 = self.up1(d2, e1 if self.use_skip else None)
        return [d1, d2, d3]


class Stage2Model(nn.Module):
    def __init__(self, use_skip=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(use_skip)

    def forward(self, x):
        enc = self.encoder(x)
        # e1..e3 are reconstruction targets -> stop-gradient (prevents a trivial
        # encoder shortcut); e4 is NOT detached so decoder gradients reach the encoder.
        targets = [e.detach() for e in enc[:3]]
        dec = self.decoder(enc)
        return targets, dec


def set_bn_momentum(model, m):
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = m


# =========================================================================
# MoCo -> encoder weights
# =========================================================================
def load_moco_into_encoder(encoder, path):
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "encoder_q", "net"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    prefixes = ("module.encoder_q.", "encoder_q.", "module.backbone.", "backbone.", "module.", "")
    rename = {"conv1.": "stem.0.", "bn1.": "stem.1."}
    target = encoder.state_dict()
    best, best_hits = None, -1
    for pre in prefixes:
        new = {}
        for k, v in sd.items():
            if not k.startswith(pre):
                continue
            k2 = k[len(pre):]
            if k2.startswith(("fc.", "head", "projector", "proj")):  # projection head not transferred
                continue
            for a, b in rename.items():
                if k2.startswith(a):
                    k2 = b + k2[len(a):]
            if k2 in target and target[k2].shape == v.shape:
                new[k2] = v
        if len(new) > best_hits:
            best, best_hits = new, len(new)
    missing = [k for k in target if k not in best]
    encoder.load_state_dict(best, strict=False)
    print(f"  MoCo weights loaded into encoder: {best_hits}/{len(target)} tensors "
          f"({len(missing)} missing)")
    if best_hits < 0.95 * len(target):
        sys.exit("ERROR: could not match the MoCo checkpoint keys. Print its keys and adapt "
                 "`prefixes`/`rename` in load_moco_into_encoder().")


# =========================================================================
# Training (Section 4.2)
# =========================================================================
def global_cos_loss(enc, dec, weights):
    """L = sum_i w_i [1 - cos(flatten(e_i), flatten(d_i))], averaged over the batch."""
    loss = 0.0
    for w, e, d in zip(weights, enc, dec):
        loss = loss + w * (1 - F.cosine_similarity(e.flatten(1), d.flatten(1), dim=1)).mean()
    return loss


def train_stage2(model, train_items, args, device):
    loader = DataLoader(ImageList(train_items), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=len(train_items) > args.batch_size)
    opt = torch.optim.AdamW([
        {"params": model.decoder.parameters(), "lr": args.lr_dec},
        {"params": model.encoder.parameters(), "lr": args.lr_enc},
    ], weight_decay=args.wd)
    model.to(device).train()
    it, data_iter = 0, iter(loader)
    while it < args.iters:
        try:
            x, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); x, _, _ = next(data_iter)
        frozen = it < args.warmup                      # decoder warm-up: encoder frozen
        for p in model.encoder.parameters():
            p.requires_grad_(not frozen)
        model.encoder.train(not frozen)
        enc, dec = model(x.to(device))
        loss = global_cos_loss(enc, dec, args.weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        it += 1
        if it % 100 == 0 or it == 1:
            print(f"  iter {it:4d}/{args.iters}  loss {loss.item():.4f}  "
                  f"{'(encoder frozen)' if frozen else ''}")
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    return model


# =========================================================================
# Checkpoint discovery
# =========================================================================
def find_ckpt(args, seed):
    pat = args.ckpt_pattern.format(dataset=args.dataset, seed=seed)
    hits = sorted(glob.glob(os.path.join(args.ckpt_dir, pat)))
    if not hits:  # looser search: any file mentioning dataset and seed
        hits = [f for f in glob.glob(os.path.join(args.ckpt_dir, "**", "*.pt*"), recursive=True)
                if args.dataset in f.lower() and str(seed) in os.path.basename(f)]
    return hits[0] if hits else None


def load_stage2(model, path):
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    n = len(model.state_dict())
    print(f"  loaded {path}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)} (of {n})")
    if len(res.missing_keys) > 0.05 * n:
        sys.exit("ERROR: this checkpoint was saved from YOUR model class, not this reference model.\n"
                 "  -> Load it with your own code and call dump_scores.dump(...) directly\n"
                 "     (see the docstring of dump_scores.py), or use --retrain.")


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["busi", "lungct"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--moco_ckpt", default=None, help="Stage-1 MoCo checkpoint (needed only for retraining)")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--ckpt_pattern", default="{dataset}_seed{seed}*.pt*")
    ap.add_argument("--out_dir", default="scores")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--split_file", default=None, help="e.g. splits/busi_seed{seed}.txt (your original split)")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_split_seed", type=int, default=0)
    ap.add_argument("--weights", type=float, nargs=3, default=[0.2, 0.3, 0.5], help="w1 w2 w3, sum to 1")
    ap.add_argument("--use_skip", action="store_true", help="skip connections encoder->decoder (check your code!)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr_dec", type=float, default=5e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--bn_momentum", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="retrain even if a checkpoint exists")
    args = ap.parse_args()
    assert abs(sum(args.weights) - 1) < 1e-6, "weights must sum to 1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + ("" if device == "cuda" else "  (CPU: inference OK, training slow)"))
    os.makedirs(args.ckpt_dir, exist_ok=True); os.makedirs(args.out_dir, exist_ok=True)

    if args.check_only:
        for s in args.seeds:
            c = find_ckpt(args, s)
            print(f"seed {s}: {'FOUND  ' + c if c else 'MISSING -> will retrain'}")
        print(f"MoCo checkpoint: {args.moco_ckpt or '(not given)'}"
              + ("  FOUND" if args.moco_ckpt and os.path.isfile(args.moco_ckpt) else ""))
        return

    for seed in args.seeds:
        print(f"\n================ {args.dataset}  seed {seed} ================")
        set_seed(seed)
        train, val, test = (load_split_file(args.split_file, seed) if args.split_file
                            else build_splits(args, seed))
        model = Stage2Model(use_skip=args.use_skip)
        set_bn_momentum(model, args.bn_momentum)

        ckpt = None if args.retrain else find_ckpt(args, seed)
        if ckpt:
            print(f"  checkpoint found -> inference only")
            load_stage2(model, ckpt)
        else:
            if not args.moco_ckpt or not os.path.isfile(args.moco_ckpt):
                sys.exit("ERROR: no Stage-2 checkpoint and no --moco_ckpt. Retrain Stage 1 first.")
            print(f"  no checkpoint -> retraining Stage 2 ({args.iters} iters, warmup {args.warmup})")
            load_moco_into_encoder(model.encoder, args.moco_ckpt)
            train_stage2(model, train, args, device)
            ckpt = os.path.join(args.ckpt_dir, f"{args.dataset}_seed{seed}.pth")
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt)
            print(f"  saved {ckpt}")

        kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dump(model, DataLoader(ImageList(val), **kw), DataLoader(ImageList(test), **kw),
             weights=tuple(args.weights),
             out_csv=os.path.join(args.out_dir, f"{args.dataset}_seed{seed}.csv"),
             device=device, level_size=128)

    print("\nDone. Next (CPU):  python eval_overall.py --dataset", args.dataset, "--std --sweep")


if __name__ == "__main__":
    main()