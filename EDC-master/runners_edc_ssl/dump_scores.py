#!/usr/bin/env python3
"""
dump_scores.py  --  export image-level anomaly scores (and per-level anomaly
maps) from a trained Stage-2 model, in the format read by eval_overall.py.

Plug-in point: `forward_features(model, x)` must return two lists
    enc = [e1, e2, e3]   (B, C_i, H_i, W_i)   encoder features
    dec = [d1, d2, d3]   (B, C_i, H_i, W_i)   reconstructed features
Edit it to match your model class (most EDC/RD4AD-style code already returns
exactly this from model(x)).

The scoring reproduces Eqs. (7)-(9) of the paper:
    P_i  = 1 - cos(e_i, d_i)            per spatial location
    P_all = sum_i w_i * Upsample(P_i)   at 256x256
    S    = max(P_all)

Example (inside your repo, after training one seed):
    from dump_scores import dump
    dump(model, val_loader, test_loader, weights=(0.2, 0.3, 0.5),
         out_csv="scores/busi_seed42.csv", device="cuda")
Loaders must yield (image_tensor, label_str, image_id); label_str is
"normal" / "benign" / "malignant" for BUSI, "normal" / "abnormal" for LungCT.
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F


def forward_features(model, x):
    out = model(x)
    enc, dec = out[0], out[1]          # <-- adapt to your model's return value
    return list(enc)[:3], list(dec)[:3]


@torch.no_grad()
def anomaly_maps(enc, dec, out_size=256):
    """Returns (B, 3, out_size, out_size) per-level maps P_i (Eq. 7)."""
    maps = []
    for e, d in zip(enc, dec):
        p = 1.0 - F.cosine_similarity(e, d, dim=1)            # (B, H_i, W_i)
        p = F.interpolate(p.unsqueeze(1), size=out_size,
                          mode="bilinear", align_corners=False)
        maps.append(p[:, 0])
    return torch.stack(maps, dim=1)


@torch.no_grad()
def image_scores(maps, weights):
    w = torch.as_tensor(weights, dtype=maps.dtype, device=maps.device).view(1, -1, 1, 1)
    p_all = (w * maps).sum(1)                                  # Eq. (8)
    return p_all.flatten(1).max(1).values                      # Eq. (9)


@torch.no_grad()
def dump(model, val_loader, test_loader, weights, out_csv,
         device="cuda", save_levels=True, level_size=64):
    """Writes out_csv and, optionally, <out_csv>_levels.npz for --sweep."""
    assert abs(sum(weights) - 1.0) < 1e-6, "reconstruction weights must sum to 1"
    model.eval().to(device)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows, ids, lv = [], [], []
    for split, loader in (("val", val_loader), ("test", test_loader)):
        for x, labels, names in loader:
            enc, dec = forward_features(model, x.to(device))
            m = anomaly_maps(enc, dec)
            s = image_scores(m, weights).cpu().numpy()
            for n, lb, sc in zip(names, labels, s):
                rows.append((n, split, lb, float(sc)))
            if save_levels:
                small = F.adaptive_avg_pool2d(m, level_size) if level_size < m.shape[-1] else m
                # NOTE: max over a pooled map is slightly lower than over the
                # full map; use level_size=256 if you want exact scores in --sweep.
                lv.append(small.half().cpu().numpy()); ids.extend(names)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["image_id", "split", "label", "score"]); wr.writerows(rows)
    if save_levels:
        np.savez_compressed(out_csv.replace(".csv", "_levels.npz"),
                            image_id=np.array(ids, dtype=object), maps=np.concatenate(lv))
    print(f"wrote {len(rows)} rows -> {out_csv}")