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
