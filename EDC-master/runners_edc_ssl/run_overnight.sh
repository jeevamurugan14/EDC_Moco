#!/bin/bash
# Overnight runs for the CVIP revision (~6-7 h). Safe to leave unattended.
cd /home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master/runners_edc_ssl || exit 1
[ -f /home/cs24d0008/EDC_SSL/moco_env/bin/activate ] && source /home/cs24d0008/EDC_SSL/moco_env/bin/activate

TS=$(date +%Y%m%d_%H%M)
OUT=overnight_$TS
mkdir -p "$OUT"
STATUS="$OUT/STATUS.txt"
MOCO=/home/cs24d0008/EDC_SSL/EDC_Improved_Weights/moco_LungCT_APTOS_BUSI_multi_v1_200ep.pth
BUSI=/home/cs24d0008/EDC_SSL/BUSI
SEEDS="42 123 456"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "=== overnight start, output folder: $OUT ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$STATUS"

# ---------- 0. backup old ablation results (scripts overwrite them) ----------
cp -r results "results_backup_$TS" 2>/dev/null && log "backup: results_backup_$TS"
mkdir -p "saved_models_backup_$TS"
cp -r saved_models/ablation_busi_* "saved_models_backup_$TS/" 2>/dev/null && log "backup: saved_models_backup_$TS"

# ---------- 1. apply the per-subset patch ----------
cat > patch_subsets.py <<'PYEOF'
import shutil, sys
p = "../methods/edc_revised.py"
src = open(p).read()
if "[SUBSET" in src:
    sys.exit(0)
shutil.copy(p, p + ".bak")
a = "        y_true = []\n"
b = "                y_true.extend(y.cpu().tolist())\n"
c = "        y_pred = (y_prob >= thresh).astype(int)\n"
for s in (a, b, c):
    if src.count(s) != 1:
        sys.exit(f"PATCH FAILED: pattern not found exactly once: {s!r}")
src = src.replace(a, a + "        names = []\n")
src = src.replace(b, b + "                names.extend([str(n) for n in file_names])\n")
block = '''
        # ---- per-subset metrics: same model, same threshold ----
        names_l = [os.path.basename(n).lower() for n in names]
        for tag in ("benign", "malignant"):
            keep = np.array([(t == 0) or (tag in n) for t, n in zip(y_true, names_l)])
            if keep.sum() > (y_true == 0).sum():
                yt, ys, yp = y_true[keep], y_prob[keep], y_pred[keep]
                m = dict(
                    AUC=roc_auc_score(yt, ys), AP=average_precision_score(yt, ys),
                    acc=accuracy_score(yt, yp),
                    precision=float(((yp == 1) & (yt == 1)).sum() / ((yp == 1).sum() + 1e-8)),
                    recall=recall_score(yt, yp, zero_division=0),
                    f1=f1_score(yt, yp, zero_division=0),
                    specificity=specificity_score(yt, yp))
                print(f"[SUBSET {tag}] n_abn={int((yt == 1).sum())}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
'''
src = src.replace(c, c + block)
open(p, "w").write(src)
print("Patched OK")
PYEOF
python3 patch_subsets.py 2>&1 | tee -a "$STATUS"
if grep -q "SUBSET" ../methods/edc_revised.py && python3 -m py_compile ../methods/edc_revised.py; then
    PATCH_OK=1; log "patch: OK"
else
    PATCH_OK=0; log "patch: FAILED -> per-subset numbers will be missing (overall still computed)"
    grep -n "y_true = \[\]\|y_true.extend\|y_pred = (y_prob" ../methods/edc_revised.py >> "$STATUS"
fi

for f in edc_ssl_busi.py edc_ssl_busi_A1_imagenet.py edc_ssl_busi_A2_singlescale.py edc_ssl_busi_A3_frozen.py; do
    log "$f uses: $(grep -m1 -oE 'from methods\.[A-Za-z0-9_]+ import' "$f" 2>/dev/null || echo 'FILE MISSING')"
done

# ---------- helper ----------
run(){
    local name=$1; shift
    log "START $name"
    python3 -u "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    log "END   $name  (exit code $rc)"
    return $rc
}

# ---------- 2. runs (all on BUSI combined: normal vs benign+malignant) ----------
run R1_proposed edc_ssl_busi.py --data_dir "$BUSI" --moco_weights_path "$MOCO" \
    --save_name ovn_busi_proposed --seeds $SEEDS

run R2_nowarmup edc_ssl_busi.py --data_dir "$BUSI" --moco_weights_path "$MOCO" \
    --save_name ovn_busi_nowarmup --warmup_iters 0 --seeds $SEEDS

run R3_singlescale edc_ssl_busi_A2_singlescale.py --data_dir "$BUSI" \
    --moco_weights_path "$MOCO" --seeds $SEEDS

run R4_frozen edc_ssl_busi_A3_frozen.py --data_dir "$BUSI" \
    --moco_weights_path "$MOCO" --seeds $SEEDS

if ! run R5_imagenet edc_ssl_busi_A1_imagenet.py --data_dir "$BUSI" \
        --moco_weights_path "$MOCO" --seeds $SEEDS; then
    if grep -q "unrecognized arguments" "$OUT/R5_imagenet.log"; then
        log "R5 retry without --moco_weights_path"
        run R5_imagenet edc_ssl_busi_A1_imagenet.py --data_dir "$BUSI" --seeds $SEEDS
    fi
fi

# ---------- 3. summary ----------
cat > "$OUT/summarize.py" <<'PYEOF'
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
PYEOF
python3 "$OUT/summarize.py" "$OUT" >> "$STATUS" 2>&1
log "=== ALL DONE. Send me: cat $OUT/STATUS.txt ==="
