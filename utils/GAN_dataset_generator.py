import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
PATCH_AREA_RATIO = 0.20  # max ratio of bbox area covered by square patch
MIN_S = 20               # minimum patch side length (pixels)
TEST_SET = "/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/eval_final_patched/Clean_Test"
PATCH_TYPES = [
    'Naturalistic1', 'Naturalistic2', 'Naturalistic3',
    'Naturalistic4', 'Naturalistic5', 'Naturalistic6',
    'TSEA1', 'TSEA2'
]


def is_in_test(imname):
    for p in PATCH_TYPES:
        dir_path = os.path.join(TEST_SET, p, 'images')
        if os.path.exists(os.path.join(dir_path, imname)):
            return True
    return False

def create_masks(images_dir, labels_dir, output_dir, resize=512, check_test=True):

    out_img_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    image_exts = [".jpg", ".jpeg", ".png"]
    images = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in image_exts]

    # → Add progress bar
    for img_name in tqdm(images, desc="Processing images"):

        # Skip if already in testing set
        if check_test and is_in_test(img_name):
            continue

        # Load image
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        # Load label file
        label_path = os.path.join(labels_dir, Path(img_name).stem + ".txt")
        if not os.path.exists(label_path):
            continue

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, bw, bh = map(float, parts)

                if int(cls) != 0:
                    continue  # only create masks for class 0

                # Convert YOLO → pixel coordinates
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h

                xmin = int(x_center - bw / 2)
                ymin = int(y_center - bh / 2)
                xmax = int(x_center + bw / 2)
                ymax = int(y_center + bh / 2)

                # Mask size
                bbox_area = bw * bh
                max_mask_area = PATCH_AREA_RATIO * bbox_area
                max_mask_size = int(np.sqrt(max_mask_area))

                mask_size = max(MIN_S, max_mask_size)

                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)

                half = mask_size // 2

                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, cx + half)
                y2 = min(h, cy + half)

                mask[y1:y2, x1:x2] = 255

        # Resize image + mask
        resized_image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (resize, resize), interpolation=cv2.INTER_NEAREST)

        # Save
        out_img_path = os.path.join(out_img_dir, Path(img_name).stem + ".jpg")
        out_mask_path = os.path.join(out_mask_dir, Path(img_name).stem + ".png")

        cv2.imwrite(out_img_path, resized_image)
        cv2.imwrite(out_mask_path, resized_mask)


if __name__ == "__main__":
    create_masks(
        images_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/images/train/",
        labels_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/labels/train/",
        output_dir="/mnt/c/Adrianov/Projects/Project-Satanael/reconstruction/experiments/data/"
    )
    create_masks(
        images_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/images/val/",
        labels_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/labels/val/",
        output_dir="/mnt/c/Adrianov/Projects/Project-Satanael/reconstruction/experiments/data/"
    )
    create_masks(
        images_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/images/test/",
        labels_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/labels/test/",
        output_dir="/mnt/c/Adrianov/Projects/Project-Satanael/reconstruction/experiments/data/"
    )
