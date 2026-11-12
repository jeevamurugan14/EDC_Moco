#!/usr/bin/env python3
"""
eval_overall.py  --  Reviewer-requested evaluation for the CVIP 2026 revision.

Addresses:
  * R1-1  "Report an Overall result using Normal vs Benign + Malignant"
  * R2-1  "Report malignant AUROC / recall honestly" (adds specificity, prevalence)
  * Table 7 confusion matrices for every subset

KEY RULE: one model, ONE threshold per seed.  The threshold is the 95th
percentile of the *validation-normal* scores and is applied unchanged to
every BUSI subset (overall / benign / malignant).  The normal test images are
therefore scored identically in all three subsets, so the Normal row of the
confusion matrix is identical across them.

Input (one CSV per dataset and seed), e.g.  scores/busi_seed42.csv
    image_id,split,label,score
    normal (1).png,val,normal,0.4132
    benign (12).png,test,benign,0.6610
    ...
  split  : "val" or "test"            (val rows must be normal images)
  label  : "normal", "benign", "malignant"  (BUSI)  or  "normal", "abnormal" (LungCT)
  score  : image-level anomaly score  S = max(P_all)   (Eq. 9 of the paper)

Optional per-level maps (for the reconstruction-weight sensitivity study, R1-3):
    scores/busi_seed42_levels.npz  with keys  image_id, maps  (N, 3, H, W)
  produced by dump_scores.py.  Use --sweep to re-score with other weights.

Usage
    python eval_overall.py --scores scores --dataset busi   --seeds 42 123 456
    python eval_overall.py --scores scores --dataset lungct --seeds 42 123 456
    python eval_overall.py --scores scores --dataset busi --seeds 42 123 456 --sweep
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np


# --------------------------------------------------------------------------
# Metrics (numpy only; verified against scikit-learn in selftest())
# --------------------------------------------------------------------------
def auroc(y, s):
    """Mann-Whitney AUROC with average ranks for ties."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y, s):
    """Step-wise AP, identical to sklearn.metrics.average_precision_score."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(y) - 1]
    tp = np.cumsum(y)[idx]
    fp = (idx + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def threshold_metrics(y, s, t):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s, dtype=float) >= t          # Eq. (10): abnormal if S >= T
    tp = int((pred & y).sum()); fn = int((~pred & y).sum())
    fp = int((pred & ~y).sum()); tn = int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(
        Accuracy=(tp + tn) / len(y), Precision=prec, Recall=rec,
        Specificity=spec, F1=f1, BalancedAcc=(rec + spec) / 2,
        TP=tp, FN=fn, FP=fp, TN=tn,
        # precision of a trivial "everything is abnormal" classifier
        Prevalence=y.mean(),
    )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_scores(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_id"], r["split"].strip().lower(),
                         r["label"].strip().lower(), float(r["score"])))
    return rows


SUBSETS = {
    "busi": {
        "Overall (B+M)": {"benign", "malignant"},
        "Benign": {"benign"},
        "Malignant": {"malignant"},
    },
    "lungct": {"Overall": {"abnormal"}},
}


def evaluate_seed(rows, dataset, pct=95.0):
    val = np.array([s for _, sp, lb, s in rows if sp == "val"])
    bad = [lb for _, sp, lb, _ in rows if sp == "val" and lb != "normal"]
    if bad:
        sys.exit(f"ERROR: validation split contains non-normal images: {set(bad)}")
    if len(val) == 0:
        sys.exit("ERROR: no validation rows -> cannot set a validation-only threshold")
    t = float(np.percentile(val, pct))

    test = [(lb, s) for _, sp, lb, s in rows if sp == "test"]
    normal = [s for lb, s in test if lb == "normal"]
    out = {"threshold": t}
    for name, abn in SUBSETS[dataset].items():
        ab = [s for lb, s in test if lb in abn]
        y = np.r_[np.zeros(len(normal)), np.ones(len(ab))]
        s = np.r_[normal, ab]
        m = threshold_metrics(y, s, t)
        m["AUROC"] = auroc(y, s)
        m["AP"] = average_precision(y, s)
        m["n_normal"], m["n_abnormal"] = len(normal), len(ab)
        out[name] = m
    return out


# --------------------------------------------------------------------------
# Aggregation / printing
# --------------------------------------------------------------------------
MAIN = ["AUROC", "AP", "Accuracy", "Precision", "Recall", "F1"]
EXTRA = ["Specificity", "BalancedAcc", "Prevalence"]


def aggregate(per_seed, subset):
    keys = MAIN + EXTRA + ["TP", "FN", "FP", "TN"]
    arr = {k: np.array([r[subset][k] for r in per_seed], dtype=float) for k in keys}
    return {k: (v.mean(), v.std(ddof=0)) for k, v in arr.items()}


def latex_row(label, agg, keys=MAIN, std=False):
    cells = []
    for k in keys:
        mu, sd = agg[k]
        cells.append(f"{mu:.4f}" + (f" $\\pm$ {sd:.4f}" if std else ""))
    return f"{label} & " + " & ".join(cells) + r" \\"


def confusion_rows(label, agg):
    tn, fp, fn, tp = (agg[k][0] for k in ("TN", "FP", "FN", "TP"))
    n_norm, n_abn = tn + fp, tp + fn
    return [
        f"{label} & Normal   & {tn / n_norm:.2f} & {fp / n_norm:.2f} \\\\",
        f"{label} & Abnormal & {fn / n_abn:.2f} & {tp / n_abn:.2f} \\\\",
    ]


# --------------------------------------------------------------------------
# Reconstruction-weight sensitivity (inference-time re-scoring, R1-3)
# --------------------------------------------------------------------------
def sweep(args, rows_by_seed):
    grid = [w for w in itertools.product(np.arange(0.1, 0.81, 0.1), repeat=3)
            if abs(sum(w) - 1.0) < 1e-6]
    print("\nReconstruction-weight sensitivity (inference-time re-scoring only;")
    print("the decoder is NOT retrained, so this isolates the aggregation step).")
    print("Weights are reported, never selected, on the test set.\n")
    print(f"{'w1':>5}{'w2':>5}{'w3':>5}  " + "  ".join(f"{n:>16}" for n in SUBSETS[args.dataset]))
    for w in grid:
        res = []
        for seed in args.seeds:
            z = np.load(os.path.join(args.scores, f"{args.dataset}_seed{seed}_levels.npz"),
                        allow_pickle=True)
            ids, maps = list(z["image_id"]), z["maps"].astype(np.float32)
            score = {i: float((np.tensordot(np.array(w), m, axes=1)).max())
                     for i, m in zip(ids, maps)}
            new = [(i, sp, lb, score[i]) for i, sp, lb, _ in rows_by_seed[seed]]
            res.append(evaluate_seed(new, args.dataset, args.percentile))
        cells = [f"{np.mean([r[n]['AUROC'] for r in res]):.4f}" for n in SUBSETS[args.dataset]]
        print(f"{w[0]:5.1f}{w[1]:5.1f}{w[2]:5.1f}  " + "  ".join(f"{c:>16}" for c in cells))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="scores")
    ap.add_argument("--dataset", choices=["busi", "lungct"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--std", action="store_true", help="print mean +/- std in LaTeX rows")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", default=None, help="optional path to dump all numbers")
    args = ap.parse_args()

    rows_by_seed, per_seed = {}, []
    for seed in args.seeds:
        p = os.path.join(args.scores, f"{args.dataset}_seed{seed}.csv")
        rows_by_seed[seed] = read_scores(p)
        r = evaluate_seed(rows_by_seed[seed], args.dataset, args.percentile)
        per_seed.append(r)
        print(f"seed {seed}: threshold T = {r['threshold']:.5f}")

    print(f"\n=== {args.dataset.upper()} : mean over seeds {args.seeds} "
          f"(single validation-only threshold, {args.percentile:.0f}th pct) ===")
    hdr = MAIN + EXTRA
    print(f"{'subset':16}" + "".join(f"{h:>12}" for h in hdr))
    aggs = {}
    for name in SUBSETS[args.dataset]:
        aggs[name] = aggregate(per_seed, name)
        print(f"{name:16}" + "".join(f"{aggs[name][h][0]:12.4f}" for h in hdr))

    print("\n--- LaTeX rows for Table 2 / Table 4 (AUROC AP Acc Prec Rec F1) ---")
    for name in SUBSETS[args.dataset]:
        print(latex_row(f"{args.dataset.upper()} -- {name}", aggs[name], std=args.std))
    print("\n--- LaTeX rows for Table 7 (row-normalised confusion matrix) ---")
    for name in SUBSETS[args.dataset]:
        for line in confusion_rows(name, aggs[name]):
            print(line)

    if args.dataset == "busi":
        # Sanity check the reviewers will implicitly do: the normal row must be
        # identical for every subset because one threshold is used.
        spec = {n: round(aggs[n]["Specificity"][0], 6) for n in aggs}
        if len(set(spec.values())) != 1:
            print("\nWARNING: specificity differs across subsets -> thresholds are not shared!")
        else:
            print(f"\nCheck passed: identical specificity across BUSI subsets "
                  f"({next(iter(spec.values())):.4f}).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"per_seed": per_seed,
                       "mean_std": {n: {k: list(v) for k, v in a.items()} for n, a in aggs.items()}},
                      f, indent=2)
    if args.sweep:
        sweep(args, rows_by_seed)


def selftest():
    """python eval_overall.py --selftest : compares metrics with scikit-learn."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(5, 300)
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        s = np.round(rng.normal(size=n) + y * rng.uniform(0, 2), rng.integers(1, 4))
        assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-9
        assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-9
    print("selftest OK: AUROC and AP match scikit-learn")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()