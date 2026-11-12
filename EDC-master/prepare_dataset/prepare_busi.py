import os
import cv2
import random

# --------------------------------------------------
# SOURCE DATASET PATH
# --------------------------------------------------
source_dir = "/home/cs24d0008/EDC_SSL/Breast BUSI/Dataset_BUSI_with_GT"

# --------------------------------------------------
# TARGET DATASET PATH
# --------------------------------------------------
target_dir = "/home/cs24d0008/EDC_SSL/BUSI"

# --------------------------------------------------
# Create required folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "masks"), exist_ok=True)

random.seed(1)

# --------------------------------------------------
# Dataset folders
# --------------------------------------------------
# FIX (reviewer-requested protocol change):
# Benign images are NOT normal tissue -- they are real, biopsied masses
# that were confirmed non-cancerous. Using them as NORMAL training data
# taught the model to treat real lesions as the "healthy" distribution,
# and excluding them from evaluation entirely made the test-time task
# artificially easy (malignant-only anomalies are the most visually
# distinct case). Both benign and malignant are now treated as ABNORMAL
# during evaluation. Only the `normal` folder is used for training,
# consistent with a standard one-class anomaly detection setup.
# --------------------------------------------------
normal_dir    = os.path.join(source_dir, "normal")
benign_dir    = os.path.join(source_dir, "benign")
malignant_dir = os.path.join(source_dir, "malignant")

# --------------------------------------------------
# Verify folders exist
# --------------------------------------------------
for d, name in [(normal_dir, "normal"), (benign_dir, "benign"), (malignant_dir, "malignant")]:
    if not os.path.exists(d):
        raise Exception(f"❌ {name} folder not found: {d}")

normal_paths   = []
abnormal_paths = []

# --------------------------------------------------
# Load NORMAL images (normal folder ONLY)
# --------------------------------------------------
for file in os.listdir(normal_dir):
    if "_mask" in file:
        continue
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        normal_paths.append(os.path.join(normal_dir, file))

print(f"Total NORMAL (training pool): {len(normal_paths)}")

# --------------------------------------------------
# Load ABNORMAL images: BENIGN + MALIGNANT combined
# FIX: benign is no longer folded into NORMAL, and is no longer
# excluded from evaluation -- it is now part of ABNORMAL along with
# malignant, matching the standard clinically-realistic BUSI protocol.
# --------------------------------------------------
benign_count = 0
for file in os.listdir(benign_dir):
    if "_mask" in file:
        continue
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path  = os.path.join(benign_dir, file)
    base      = os.path.splitext(file)[0]
    mask_path = os.path.join(benign_dir, base + "_mask.png")
    if os.path.exists(mask_path):
        abnormal_paths.append((img_path, mask_path, "benign"))
        benign_count += 1

print(f"Benign images (→ABNORMAL)   : {benign_count}")

malignant_count = 0
for file in os.listdir(malignant_dir):
    if "_mask" in file:
        continue
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path  = os.path.join(malignant_dir, file)
    base      = os.path.splitext(file)[0]
    mask_path = os.path.join(malignant_dir, base + "_mask.png")
    if os.path.exists(mask_path):
        abnormal_paths.append((img_path, mask_path, "malignant"))
        malignant_count += 1

print(f"Malignant images (→ABNORMAL): {malignant_count}")
print(f"Total ABNORMAL (benign+malignant): {len(abnormal_paths)}")

# --------------------------------------------------
# Print raw dataset statistics
# --------------------------------------------------
print("\n========== RAW DATASET INFO ==========")
print("Total normal images   :", len(normal_paths))
print("Total abnormal images :", len(abnormal_paths),
      f"({benign_count} benign + {malignant_count} malignant)")

# --------------------------------------------------
# Split normal 80% train / 20% test
# --------------------------------------------------
random.shuffle(normal_paths)
split         = int(len(normal_paths) * 0.8)
train_normals = normal_paths[:split]
test_normals  = normal_paths[split:]

# Use ALL abnormal (benign + malignant) for test (unbalanced — more reliable AUC)
test_abnormals = abnormal_paths

print(f"\n[UNBALANCED] Using ALL {len(test_abnormals)} ABNORMAL for test "
      f"({benign_count} benign + {malignant_count} malignant)")
print(f"  Train NORMAL  : {len(train_normals)}")
print(f"  Test  NORMAL  : {len(test_normals)}")
print(f"  Test  ABNORMAL: {len(test_abnormals)}")

# --------------------------------------------------
# Create train/test folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "train", "NORMAL"),   exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "NORMAL"),   exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "ABNORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "ABNORMAL_BENIGN"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "ABNORMAL_MALIGNANT"), exist_ok=True)

# --------------------------------------------------
# Resize and save helper
# --------------------------------------------------
def process_and_save(src, dst, is_mask=False):
    if is_mask:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(src)
    if img is None:
        print(f"❌ Could not read: {src}")
        return
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    img = cv2.resize(img, (256, 256), interpolation=interpolation)
    cv2.imwrite(dst, img)

# --------------------------------------------------
# Save training NORMAL images (normal folder only)
# --------------------------------------------------
for path in train_normals:
    name = os.path.basename(path)
    dst  = os.path.join(target_dir, "train", "NORMAL", name)
    process_and_save(path, dst)

# --------------------------------------------------
# Save testing NORMAL images
# --------------------------------------------------
for path in test_normals:
    name = os.path.basename(path)
    dst  = os.path.join(target_dir, "test", "NORMAL", name)
    process_and_save(path, dst)

# --------------------------------------------------
# Save ALL ABNORMAL images + masks (benign + malignant)
# --------------------------------------------------
for img_path, mask_path, cls in test_abnormals:
    name     = os.path.basename(img_path)
    base     = os.path.splitext(name)[0]
    # prefix images with class label to avoid filename collisions
    # and to make each image's class unambiguous from its filename alone
    name = f"{cls}__" + name
    base = f"{cls}__" + base

    subclass_folder = "ABNORMAL_BENIGN" if cls == "benign" else "ABNORMAL_MALIGNANT"

    dst_combined = os.path.join(target_dir, "test", "ABNORMAL", name)
    dst_subclass = os.path.join(target_dir, "test", subclass_folder, name)
    mask_dst     = os.path.join(target_dir, "masks", base + "_mask.png")

    process_and_save(img_path, dst_combined)
    process_and_save(img_path, dst_subclass)
    process_and_save(mask_path, mask_dst, is_mask=True)

# --------------------------------------------------
# Final summary
# --------------------------------------------------
print("\n========== FINAL DATASET ==========")
print("Train NORMAL  :", len(train_normals))
print("Test  NORMAL  :", len(test_normals))
print("Test  ABNORMAL:", len(test_abnormals),
      f"({benign_count} benign + {malignant_count} malignant)")
print("Masks         :", len(test_abnormals))
print(f"\ntest/ABNORMAL           : {len(test_abnormals)} (combined benign+malignant)")
print(f"test/ABNORMAL_BENIGN    : {benign_count}")
print(f"test/ABNORMAL_MALIGNANT : {malignant_count}")
print(f"\nClass ratio (test) : "
      f"NORMAL {len(test_normals)} : ABNORMAL {len(test_abnormals)} "
      f"= 1 : {len(test_abnormals)/max(len(test_normals),1):.2f}")
print("\n✅ Dataset preparation complete (benign+malignant combined as ABNORMAL).")