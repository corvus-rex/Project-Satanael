import cv2
import numpy as np
from pathlib import Path
import random
import math
import shutil
from tqdm import tqdm 

# Configuration
PATCH_AREA_RATIO = 0.20
MIN_S = 80

patch_types = [
    'Naturalistic1', 'Naturalistic2', 'Naturalistic3',
    'Naturalistic4', 'Naturalistic5', 'Naturalistic6',
    'TSEA1', 'TSEA2'
]

img_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\images\merged")
label_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\labels\merged")
patch_root = Path(r"C:\Adrianov\Projects\Project-Satanael\adv_patches")
out_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\eval_final_patched")

# Preload patches
patches = {}
for ptype in patch_types:
    ppath = patch_root / f"{ptype}.png"
    patch_img = cv2.imread(str(ppath), cv2.IMREAD_UNCHANGED)
    if patch_img is None:
        raise FileNotFoundError(f"Patch not found at {ppath}")
    patches[ptype] = patch_img

# Prepare output directories
for ptype in patch_types:
    (out_dir / ptype / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / ptype / "labels").mkdir(parents=True, exist_ok=True)

# First pass: collect image info
image_infos = []
for img_path in tqdm(list(img_dir.glob("*.jpg")), desc="Scanning images"):
    label_path = label_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    patchable_boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_center, y_center, bw, bh = map(float, parts)
        if int(cls) != 0:
            continue
        box_w, box_h = int(bw * w), int(bh * h)
        max_patch_area = int(PATCH_AREA_RATIO * box_w * box_h)
        patch_size = int(math.sqrt(max_patch_area))
        if patch_size >= MIN_S:
            patchable_boxes.append((x_center, y_center, bw, bh))

    image_infos.append({
        "img_path": img_path,
        "label_path": label_path,
        "size": (w, h),
        "lines": lines,
        "patchable_boxes": patchable_boxes
    })

# Separate patchable/unpatchable
patchable = [i for i in image_infos if i["patchable_boxes"]]
unpatchable = [i for i in image_infos if not i["patchable_boxes"]]

# Distribute patchable images equally
count_per_patch = math.ceil(len(patchable) / len(patch_types))
ptype_cycle = [ptype for ptype in patch_types for _ in range(count_per_patch)]
random.shuffle(patchable)
assignments = list(zip(patchable, ptype_cycle))

# Process patchable images
for info, chosen_type in tqdm(assignments, desc="Applying patches"):
    img_path = info["img_path"]
    label_path = info["label_path"]
    img = cv2.imread(str(img_path))
    w, h = info["size"]
    lines = info["lines"]
    new_labels = lines.copy()
    patch = patches[chosen_type]
    patched = False

    for x_center, y_center, bw, bh in info["patchable_boxes"]:
        xc, yc = int(x_center * w), int(y_center * h)
        box_w, box_h = int(bw * w), int(bh * h)
        max_patch_area = int(PATCH_AREA_RATIO * box_w * box_h)
        patch_size = int(math.sqrt(max_patch_area))
        if patch_size < MIN_S:
            continue

        resized_patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        x1, y1 = xc - patch_size // 2, yc - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size

        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, w), min(y2, h)
        region = img[y1c:y2c, x1c:x2c]

        patch_crop = resized_patch[
            y1c - y1: patch_size - (y2 - y2c),
            x1c - x1: patch_size - (x2 - x2c)
        ]
        if region.shape[:2] != patch_crop.shape[:2]:
            continue

        if patch_crop.shape[2] == 4:
            alpha = patch_crop[:, :, 3] / 255.0
            for c in range(3):
                region[:, :, c] = region[:, :, c] * (1 - alpha) + patch_crop[:, :, c] * alpha
        else:
            region[:] = patch_crop
        img[y1c:y2c, x1c:x2c] = region
        patched = True

        # Patch bbox label (class 1)
        patch_cx = ((x1c + x2c) / 2) / w
        patch_cy = ((y1c + y2c) / 2) / h
        patch_bw = (x2c - x1c) / w
        patch_bh = (y2c - y1c) / h
        new_labels.append(f"1 {patch_cx:.6f} {patch_cy:.6f} {patch_bw:.6f} {patch_bh:.6f}")

    if patched:
        new_name = f"{chosen_type}_{img_path.name}"
        new_label_name = f"{chosen_type}_{label_path.name}"
        cv2.imwrite(str(out_dir / chosen_type / "images" / new_name), img)
        with open(out_dir / chosen_type / "labels" / new_label_name, "w") as f:
            f.write("\n".join(new_labels) + "\n")

# Handle unpatchable
for info in tqdm(unpatchable, desc="Copying unpatchable"):
    dest_img = out_dir / "unpatched" / "images"
    dest_lbl = out_dir / "unpatched" / "labels"
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)
    shutil.copy(info["img_path"], dest_img / info["img_path"].name)
    shutil.copy(info["label_path"], dest_lbl / info["label_path"].name)

print("Balanced patched dataset created with grouping, renaming, and progress bars.")
