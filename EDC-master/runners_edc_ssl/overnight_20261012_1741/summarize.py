import re, sys, glob, os, numpy as np
out_dir = sys.argv[1]
pat = re.compile(r"\[SUBSET (benign|malignant)\] n_abn=(\d+)\s+(.*)")
kv = re.compile(r"(\w+)=([0-9.]+)")
lines_out = []
for f in sorted(glob.glob(os.path.join(out_dir, "*.log"))):
    last, per_seed = {}, {}
    for line in open(f, errors="ignore"):
        m = pat.search(line)
        if m:
            last[m.group(1)] = {k: float(v) for k, v in kv.findall(m.group(3))}
            continue
        if "Seed" in line and ("Final Results" in line or line.strip().startswith("BUSI |")):
            s = re.search(r"Seed (\d+)", line)
            if s and last:
                per_seed[s.group(1)] = dict(last)
    lines_out.append(f"\n===== {os.path.basename(f)} =====  seeds found: {sorted(per_seed)}")
    for tag in ("benign", "malignant"):
        rows = [per_seed[s][tag] for s in per_seed if tag in per_seed[s]]
        if rows:
            keys = rows[0].keys()
            lines_out.append(f"{tag:9s} (mean of {len(rows)}): " +
                "  ".join(f"{k}={np.mean([r[k] for r in rows]):.4f}" for k in keys))
    summ = [l.rstrip() for l in open(f, errors="ignore")
            if re.match(r"\s+(eval/)?(AUC|AUROC|AP|acc|Accuracy|f1|F1|recall|Recall|precision|Precision|specificity|Specificity)\s*:", l)]
    lines_out += ["overall summary:"] + summ[-7:]
txt = "\n".join(lines_out)
print(txt)
open(os.path.join(out_dir, "SUMMARY.txt"), "w").write(txt + "\n")
