"""
check_patient_overlap.py
A7: Verify no patient/scan appears in more than one split (train/val/test).

Run this from your project root:
    python check_patient_overlap.py --dataset busi
    python check_patient_overlap.py --dataset lungct

WHAT THIS DOES
--------------
1. Lists every filename actually used in your train / val / test splits
   (using the SAME AD_Dataset_WithVal class your training scripts use,
   so this reflects your real splits, not a guess).
2. Extracts a "patient/scan ID" from each filename using a regex you
   may need to adjust (see PATIENT_ID_PATTERNS below -- I've guessed
   at common conventions, but BUSI/LungCT filenames vary by source).
3. Reports:
   - How many unique patient IDs are in each split
   - Whether ANY patient ID appears in more than one split (this is
     the leakage check -- the answer must be "no overlap" for your
     splits to be safe)

IF THE DEFAULT PATTERN DOESN'T MATCH YOUR FILENAMES:
Print a few example filenames first (see the --dry_run flag) and send
them to me -- I'll write the correct regex for your actual naming
convention in one line.
"""

import argparse
import re
from collections import defaultdict

from datasets.dataset_with_val import AD_Dataset_WithVal


# ------------------------------------------------------------------
# GUESSED patient-ID patterns -- adjust these to match your real
# filenames if the dry run below shows a mismatch.
# ------------------------------------------------------------------
PATIENT_ID_PATTERNS = {
    # BUSI convention is usually like "benign (12).png", "malignant (7).png",
    # "normal (3).png" -- the number in parentheses is the case/patient index.
    "busi": re.compile(r"\((\d+)\)"),

    # LungCT / generic CT slice naming is often like "patient012_slice034.png"
    # or "P012_S034.png" -- adjust prefix/pattern as needed.
    "lungct": re.compile(r"(patient\d+|P\d+|pid\d+)", re.IGNORECASE),
}


def get_filenames(dataset_name, data_dir, img_size=256, seed=42):
    """Returns dict: split_name -> list of filenames actually used in that split."""
    ds = AD_Dataset_WithVal(
        name=dataset_name, data_dir=data_dir,
        img_size=img_size, crop_size=img_size,
        val_fraction=0.2, seed=seed, train=True,
    )
    train_dset = ds.get_train_dset()
    val_dset = ds.get_val_dset()
    eval_dset = AD_Dataset_WithVal(
        name=dataset_name, data_dir=data_dir, img_size=img_size, train=False,
    ).get_eval_dset()

    def extract_names(dset):
        # Most datasets expose .samples (list of (path, label)) or .imgs;
        # fall back to .data / .filenames if your dataset class differs.
        if hasattr(dset, "samples"):
            return [s[0] for s in dset.samples]
        elif hasattr(dset, "imgs"):
            return [s[0] for s in dset.imgs]
        elif hasattr(dset, "filenames"):
            return list(dset.filenames)
        else:
            raise AttributeError(
                "Could not find filenames on this dataset object. "
                "Tell me the attribute name that holds file paths and I'll fix this."
            )

    return {
        "train": extract_names(train_dset),
        "val": extract_names(val_dset),
        "test": extract_names(eval_dset),
    }


def extract_patient_id(filename, pattern):
    match = pattern.search(filename)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["busi", "lungct", "aptos"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                         help="Just print 10 example filenames per split and exit, "
                              "so you can check the regex pattern matches.")
    args = parser.parse_args()

    filenames = get_filenames(args.dataset, args.data_dir, args.img_size, args.seed)

    if args.dry_run:
        print(f"\n=== DRY RUN: example filenames for '{args.dataset}' ===")
        for split, names in filenames.items():
            print(f"\n--- {split} ({len(names)} files) ---")
            for n in names[:10]:
                print(f"  {n}")
        print("\nCheck these against PATIENT_ID_PATTERNS in this script.")
        print("If the pattern won't match, send me 10-20 example filenames and I'll fix the regex.")
        return

    pattern = PATIENT_ID_PATTERNS.get(args.dataset)
    if pattern is None:
        print(f"No pattern defined for '{args.dataset}'. Run with --dry_run first, "
              f"send me example filenames, and I'll add one.")
        return

    split_patient_ids = {}
    unmatched_count = {}
    for split, names in filenames.items():
        ids = set()
        unmatched = 0
        for n in names:
            pid = extract_patient_id(n, pattern)
            if pid is None:
                unmatched += 1
            else:
                ids.add(pid)
        split_patient_ids[split] = ids
        unmatched_count[split] = unmatched

    print(f"\n=== Patient/scan ID summary for '{args.dataset}' ===")
    for split in ["train", "val", "test"]:
        print(f"  {split:6s}: {len(filenames[split]):5d} files, "
              f"{len(split_patient_ids[split]):5d} unique IDs, "
              f"{unmatched_count[split]:5d} filenames didn't match the pattern")

    # ---- Overlap check: the actual leakage question ----
    print(f"\n=== Overlap check (this is the leakage answer) ===")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    any_overlap = False
    for a, b in pairs:
        overlap = split_patient_ids[a] & split_patient_ids[b]
        if overlap:
            any_overlap = True
            print(f"  ⚠ OVERLAP between {a} and {b}: {len(overlap)} shared patient IDs")
            print(f"    e.g. {list(overlap)[:10]}")
        else:
            print(f"  ✓ No overlap between {a} and {b}")

    if any_overlap:
        print("\n*** ACTION NEEDED: patient-level leakage detected. ***")
        print("This means B6 (re-split + retrain) is now required before trusting any results.")
    else:
        print("\n*** No patient-level leakage detected. Splits are patient-safe. ***")
        if any(unmatched_count.values()):
            print("NOTE: some filenames didn't match the ID pattern -- double check those")
            print("manually before fully trusting this result, in case they hide real overlaps.")


if __name__ == "__main__":
    main()