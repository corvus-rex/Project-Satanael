import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Helper: convert YOLO label format to (x1, y1, x2, y2)
def yolo_to_xyxy(yolo_label, img_w, img_h):
    cls, x, y, w, h = yolo_label
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0.0

def evaluate_map50(image_dir: str, label_dir: str, class_id: int, model) -> float:
    image_paths = list(Path(image_dir).glob("*.jpg"))
    if not image_paths:
        raise ValueError("No .jpg images found in the image_dir.")

    all_predictions = []
    all_gts = []

    i = 0
    for image_path in image_paths:
        print(i)
        base_name = image_path.stem
        label_path = Path(label_dir) / f"{base_name}.txt"
        if not label_path.exists():
            continue

        # Load image and run inference
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        results = model(image)
        preds = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        # Load GT labels and convert to [x1, y1, x2, y2]
        with open(label_path) as f:
            lines = f.read().strip().splitlines()
        gts = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if int(parts[0]) == class_id:
                gts.append(yolo_to_xyxy(parts, w, h))

        preds = [p for p in preds if int(p[5]) == class_id]

        all_predictions.extend([[*p[:4], p[4], image_path.name] for p in preds])
        all_gts.append((image_path.name, gts))
        i+= 1

    all_predictions.sort(key=lambda x: -x[4])

    tp = []
    fp = []
    matched = {}

    total_gts = sum(len(gts) for _, gts in all_gts)

    for pred in all_predictions:
        box_pred = pred[:4]
        conf = pred[4]
        img_name = pred[5]

        gts = dict(all_gts).get(img_name, [])
        ious = [iou(box_pred, gt_box) for gt_box in gts]
        max_iou = max(ious) if ious else 0
        max_idx = ious.index(max_iou) if ious else -1

        if max_iou >= 0.5:
            key = (img_name, max_idx)
            if key not in matched:
                tp.append(1)
                fp.append(0)
                matched[key] = True
            else:
                tp.append(0)
                fp.append(1)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (total_gts + 1e-6)

    ap = 0.0
    for r in np.linspace(0, 1, 101):
        precisions = precision[recall >= r]
        p = max(precisions) if len(precisions) else 0
        ap += p / 101

    return ap
