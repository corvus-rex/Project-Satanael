import os
import cv2
import numpy as np
from pathlib import Path

# Configuration
PATCH_AREA_RATIO = 0.20  # max ratio of bbox area covered by square patch
MIN_S = 60               # minimum patch side length (pixels)

def create_masks(images_dir, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_exts = [".jpg", ".jpeg", ".png"]
    images = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in image_exts]

    for img_name in images:
        # Read image to get dimensions
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        # Corresponding YOLO label file
        label_path = os.path.join(labels_dir, Path(img_name).stem + ".txt")
        if not os.path.exists(label_path):
            continue

        # Initialize blank mask
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, bw, bh = map(float, parts)

                if int(cls) != 0:
                    continue  # only mask class 0

                # Convert YOLO coords to pixel coordinates
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h

                # Bounding box corners
                xmin = int(x_center - bw / 2)
                ymin = int(y_center - bh / 2)
                xmax = int(x_center + bw / 2)
                ymax = int(y_center + bh / 2)

                # Calculate mask size
                bbox_area = bw * bh
                max_mask_area = PATCH_AREA_RATIO * bbox_area
                max_mask_size = int(np.sqrt(max_mask_area))  # since mask is square

                mask_size = max(MIN_S, max_mask_size)

                # Center of bbox
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)

                # Half mask size
                half = mask_size // 2

                # Mask coordinates (clipped to image boundaries)
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, cx + half)
                y2 = min(h, cy + half)

                # Draw mask
                mask[y1:y2, x1:x2] = 255

        # Save mask
        out_path = os.path.join(output_dir, Path(img_name).stem + ".png")
        cv2.imwrite(out_path, mask)

if __name__ == "__main__":
    # Example usage (set these paths to your dataset)
    create_masks(
        images_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/images/train/",
        labels_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/labels/train/",
        output_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/GAN_dataset/"
    )
    create_masks(
        images_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/images/val/",
        labels_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/labels/val/",
        output_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/GAN_dataset/"
    )
    create_masks(
        images_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/images/test/",
        labels_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/labels/test/",
        output_dir="/mnt/c/Adrianov/Projects/Project-Satanael/data/tju-dhd/GAN_dataset/"
    )
