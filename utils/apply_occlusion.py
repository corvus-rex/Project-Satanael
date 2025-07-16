import cv2
import numpy as np
from pathlib import Path
import shutil

# Configuration
ratio = 0.6

# Paths
img_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\images\test")
label_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\labels\test")

out_img_dir = Path(rf"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\occluded\images")
out_label_dir = Path(rf"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\occluded\labels")
out_img_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

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

        # Compute top-left coordinates for placement
        x1 = xc - patch_size // 2
        y1 = yc - patch_size // 2
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        # Clip to image bounds
        x1_clamp, y1_clamp = max(x1, 0), max(y1, 0)
        x2_clamp, y2_clamp = min(x2, w), min(y2, h)

        patch_w_clamp = x2_clamp - x1_clamp
        patch_h_clamp = y2_clamp - y1_clamp

        if patch_w_clamp <= 0 or patch_h_clamp <= 0:
            continue

        # Create black patch
        black_patch = np.zeros((patch_h_clamp, patch_w_clamp, 3), dtype=np.uint8)

        # Apply patch
        img[y1_clamp:y2_clamp, x1_clamp:x2_clamp] = black_patch

    # Save output
    out_img_path = out_img_dir / img_path.name
    out_label_path = out_label_dir / label_path.name

    cv2.imwrite(str(out_img_path), img)
    shutil.copy(str(label_path), str(out_label_path))

print("Patched test set with black boxes created.")
