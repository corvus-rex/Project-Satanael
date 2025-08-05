import os
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Paths ===
image_dir = r"C:\Adrianov\Projects\Project-Satanael\data\APRICOTv1.0\Images\test_PAD"           # folder with images
annotation_path = r"C:\Adrianov\Projects\Project-Satanael\data\APRICOTv1.0\Annotations\coco_apricot_annotations_test.json"  # JSON with annotations
output_dir = r"C:\Adrianov\Projects\Project-Satanael\data\APRICOTv1.0\Images\test_PAD_viz"         # folder to save annotated images (optional)
os.makedirs(output_dir, exist_ok=True)

# === Load JSON ===
with open(annotation_path, "r") as f:
    coco_data = json.load(f)

# === Build image_id → filename mapping ===
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# === Group annotations by image_id ===
ann_by_image = {}
for ann in coco_data["annotations"]:
    if ann["category_id"] == 12:  # ✅ Only include category 12
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

# === Process each image ===
for image_id, filename in tqdm(image_id_to_filename.items()):
    if image_id not in ann_by_image:
        continue  # Skip if no relevant annotations

    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    anns = ann_by_image[image_id]

    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        angle = ann.get("angle", ["", "", ""])[2]
        warped = ann.get("is_warped", False)

        # Draw box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label
        label = f"angle:{angle} warped:{warped}"
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save output
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print(f"✅ Done. Visualized category_id 12 in: {output_dir}")