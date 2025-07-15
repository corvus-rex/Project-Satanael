import cv2
import numpy as np
from pathlib import Path
import shutil

# Configuration
ratio = 0.75  # fraction of bbox width to use for patch size

# Paths
img_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\images\test")
label_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\labels\test")
patch_path = Path(r"C:\Adrianov\Projects\Project-Satanael\data\adversarial_patch\v5-demo.png")
# img_dir = Path("/home/ubuntu/adrian/tju-dhd/images/test")
# label_dir = Path("/home/ubuntu/adrian/tju-dhd/labels/test")
# patch_path = Path("/home/ubuntu/adrian/v5-demo.png")

# Output dirs
out_img_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\patched\images")
out_label_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\patched\labels")
# out_img_dir = Path("/home/ubuntu/adrian/tju-dhd/patched/images/test")
# out_label_dir = Path("/home/ubuntu/adrian/tju-dhd/patched/labels/test")
out_img_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

# Load patch
patch = cv2.imread(str(patch_path), cv2.IMREAD_UNCHANGED)
if patch is None:
    raise FileNotFoundError(f"Patch not found at {patch_path}")
patch_h, patch_w = patch.shape[:2]

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

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, x_center, y_center, bw, bh = map(float, parts)
        if int(cls) != 0:
            continue

        xc, yc = int(x_center * w), int(y_center * h)
        box_w, box_h = int(bw * w), int(bh * h)

        patch_size = int(box_w * ratio)

        if patch_size < 1:
            continue

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

    # Save output
    out_img_path = out_img_dir / img_path.name
    out_label_path = out_label_dir / label_path.name

    cv2.imwrite(str(out_img_path), img)
    shutil.copy(str(label_path), str(out_label_path))

print("Patched test set created.")
