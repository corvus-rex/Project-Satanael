import cv2
import numpy as np
from pathlib import Path
import shutil
import random

# Configuration
PATCH_AREA_RATIO = 0.20  # max ratio of bbox area covered by square patch
MIN_S = 60               # minimum patch side length (pixels)

# List of possible patch types
patch_types = ['Naturalistic1', 'Naturalistic2', 'Naturalistic3', 'Naturalistic4', 'Naturalistic5', 'Naturalistic6', 'TSEA1']
# patch_types = ['Naturalistic1', 'Naturalistic3', 'Naturalistic4', 'Naturalistic6', 'TSEA1']
patch_types = ['Naturalistic2', 'Naturalistic5']


# Paths
img_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\images\test")
label_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\labels\test")
patch_root = Path(r"C:\Adrianov\Projects\Project-Satanael\adv_patches")

out_img_dir = Path(rf"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\patched_random_bigger_test\images")
out_label_dir = Path(rf"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\patched_random_bigger_test\labels")
out_img_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

# Preload all patches
patches = {}
for ptype in patch_types:
    ppath = patch_root / f"{ptype}.png"
    patch_img = cv2.imread(str(ppath), cv2.IMREAD_UNCHANGED)
    if patch_img is None:
        raise FileNotFoundError(f"Patch not found at {ppath}")
    patches[ptype] = patch_img

# Process each image
for img_path in img_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    label_path = label_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    with open(label_path, "r") as f:
        lines = f.read().splitlines()

    patched = False  # track whether a patch was applied
    new_labels = lines.copy()  # keep original person labels

    # Randomly choose one patch type for this image
    chosen_type = random.choice(patch_types)
    patch = patches[chosen_type]

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, x_center, y_center, bw, bh = map(float, parts)
        if int(cls) != 0:
            continue  # only apply patch to class 0 (person)

        # Convert YOLO format to pixel coordinates
        xc, yc = int(x_center * w), int(y_center * h)
        box_w, box_h = int(bw * w), int(bh * h)

        # Compute patch size (square with area <= PATCH_AREA_RATIO of bbox)
        max_patch_area = int(PATCH_AREA_RATIO * box_w * box_h)
        patch_size = int(np.sqrt(max_patch_area))

        # Skip if patch is too small
        if patch_size < MIN_S:
            continue

        # Resize patch
        resized_patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        # Compute top-left coordinates for placement
        x1 = xc - patch_size // 2
        y1 = yc - patch_size // 2
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        # Clip to image bounds
        x1_clamp, y1_clamp = max(x1, 0), max(y1, 0)
        x2_clamp, y2_clamp = min(x2, w), min(y2, h)

        patch_x1 = x1_clamp - x1
        patch_y1 = y1_clamp - y1
        patch_x2 = patch_size - (x2 - x2_clamp)
        patch_y2 = patch_size - (y2 - y2_clamp)

        region = img[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
        patch_crop = resized_patch[patch_y1:patch_y2, patch_x1:patch_x2]

        if patch_crop.shape[:2] != region.shape[:2]:
            continue  # mismatch in size, skip

        if patch_crop.shape[2] == 4:
            # Alpha blending
            alpha = patch_crop[:, :, 3] / 255.0
            for c in range(3):
                region[:, :, c] = region[:, :, c] * (1 - alpha) + patch_crop[:, :, c] * alpha
        else:
            region[:] = patch_crop

        img[y1_clamp:y2_clamp, x1_clamp:x2_clamp] = region
        patched = True

        # Compute YOLO bbox for patch (class 1)
        patch_cx = ((x1_clamp + x2_clamp) / 2) / w
        patch_cy = ((y1_clamp + y2_clamp) / 2) / h
        patch_bw = (x2_clamp - x1_clamp) / w
        patch_bh = (y2_clamp - y1_clamp) / h

        new_labels.append(f"1 {patch_cx:.6f} {patch_cy:.6f} {patch_bw:.6f} {patch_bh:.6f}")

    # Save output only if at least one patch was applied
    if patched:
        out_img_path = out_img_dir / img_path.name
        out_label_path = out_label_dir / label_path.name

        cv2.imwrite(str(out_img_path), img)
        with open(out_label_path, "w") as f:
            f.write("\n".join(new_labels) + "\n")

print("Patched test set created (random patch per image, YOLO labels updated).")
