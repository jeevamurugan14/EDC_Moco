"""
preflight_check.py
Run this BEFORE run_all_baselines.sh to catch problems in seconds
instead of hours into a run.

Usage:
    python preflight_check.py --moco_weights_path /your/real/path.pth

Checks:
  1. All 9 script files actually exist in runners_edc_ssl/
  2. Your MoCo weights file actually exists on disk
  3. torchvision.models.resnet50 loads correctly with the
     pretrained=True fallback (the exact fix applied to PaDiM/RD4AD)
  4. methods.edc_revised.EDC and its dependencies import cleanly
     (needed by the three A6 scripts)
  5. scipy is installed (needed by PaDiM's Mahalanobis distance step)
"""

import os
import sys
import argparse


def check(description, fn):
    print(f"  {description:55s}", end="", flush=True)
    try:
        fn()
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moco_weights_path", type=str, required=True)
    parser.add_argument("--runners_dir", type=str, default="runners_edc_ssl")
    args = parser.parse_args()

    print("=" * 60)
    print("  PRE-FLIGHT CHECK")
    print("=" * 60)

    all_ok = True

    # ---- 1. All expected script files exist ----
    print("\n[1] Checking all 9 script files exist...")
    expected_files = [
        "edc_ssl_lungct_baseline_padim.py",
        "edc_ssl_lungct_baseline_rd4ad.py",
        "edc_ssl_busi_baseline_padim.py",
        "edc_ssl_busi_baseline_rd4ad.py",
        "edc_ssl_aptos_baseline_padim.py",
        "edc_ssl_aptos_baseline_rd4ad.py",
        "edc_ssl_lungct_A6_nowarmup.py",
        "edc_ssl_busi_A6_nowarmup.py",
        "edc_ssl_aptos_A6_nowarmup.py",
    ]
    for fname in expected_files:
        path = os.path.join(args.runners_dir, fname)
        ok = check(f"{fname}", lambda p=path: (_ for _ in ()).throw(
            FileNotFoundError(f"not found at {p}")) if not os.path.isfile(p) else None)
        all_ok = all_ok and ok

    # ---- 2. MoCo weights file actually exists ----
    print("\n[2] Checking MoCo weights file...")
    ok = check(f"{args.moco_weights_path}", lambda: (_ for _ in ()).throw(
        FileNotFoundError("file does not exist -- check the path")) if not os.path.isfile(args.moco_weights_path) else None)
    all_ok = all_ok and ok

    # ---- 3. torchvision resnet50 loads with the fallback fix ----
    print("\n[3] Checking torchvision.models.resnet50 loading...")
    def _load_resnet():
        import torchvision.models as tv_models
        try:
            _ = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            print(" [new-API path used] ", end="")
        except AttributeError:
            _ = tv_models.resnet50(pretrained=True)
            print(" [legacy pretrained=True path used] ", end="")
    ok = check("resnet50 pretrained load", _load_resnet)
    all_ok = all_ok and ok

    # ---- 4. methods.edc_revised imports cleanly (needed by A6 scripts) ----
    print("\n[4] Checking methods.edc_revised imports...")
    def _import_edc_revised():
        from methods.edc_revised import EDC  # noqa: F401
    ok = check("from methods.edc_revised import EDC", _import_edc_revised)
    all_ok = all_ok and ok

    def _import_train_utils():
        from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup  # noqa: F401
    ok = check("train_utils imports", _import_train_utils)
    all_ok = all_ok and ok

    def _import_moco_model():
        from models.edc_ssl import MoCo_R50_R50  # noqa: F401
    ok = check("models.edc_ssl.MoCo_R50_R50", _import_moco_model)
    all_ok = all_ok and ok

    # ---- 5. scipy available (needed for PaDiM's Mahalanobis distance) ----
    print("\n[5] Checking scipy is installed...")
    def _import_scipy():
        from scipy.spatial.distance import mahalanobis  # noqa: F401
    ok = check("scipy.spatial.distance.mahalanobis", _import_scipy)
    all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED -- safe to run run_all_baselines.sh")
    else:
        print("  SOME CHECKS FAILED -- fix the items above before running")
        print("  the full script, or you'll hit these same errors hours in.")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()