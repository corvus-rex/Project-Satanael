import json
import os
import math
import random
import yaml
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANN_DIR = os.path.join(ROOT_DIR, 'data', 'dhd_pedestrian_traffic_trainval_annos', 'dhd_pedestrian', 'ped_traffic', 'annotations')
IMG_DIR = os.path.join(ROOT_DIR, 'data', 'dhd_traffic_trainval_images')

IMG_TRAIN_DIR = os.path.join(IMG_DIR, 'dhd_traffic', 'images', 'train')
IMG_VAL_DIR = os.path.join(IMG_DIR, 'dhd_traffic', 'images', 'val')
ANN_TRAIN_DIR = os.path.join(ANN_DIR, 'dhd_pedestrian_traffic_train.json')
ANN_VAL_DIR = os.path.join(ANN_DIR, 'dhd_pedestrian_traffic_val.json')

YOLO_DIR = os.path.join(ROOT_DIR, 'data', 'yolov5')
YOLO_IMG_TRAIN = os.path.join(YOLO_DIR, 'images', 'train')
YOLO_IMG_VAL = os.path.join(YOLO_DIR, 'images', 'val')
YOLO_IMG_TEST = os.path.join(YOLO_DIR, 'images', 'test')
YOLO_LABEL_TRAIN = os.path.join(YOLO_DIR, 'labels', 'train')
YOLO_LABEL_VAL = os.path.join(YOLO_DIR, 'labels', 'val')
YOLO_LABEL_TEST = os.path.join(YOLO_DIR, 'labels', 'test')

def yolo_label(id, annotations, w, h, type):
    lines = []
    if type == 'train':
        label_path = os.path.join(YOLO_LABEL_TRAIN, f'{id}.txt')
    elif type == 'val':
        label_path = os.path.join(YOLO_LABEL_VAL, f'{id}.txt')
    elif type == 'test':
        label_path = os.path.join(YOLO_LABEL_TEST, f'{id}.txt')
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    for bbox in annotations:
        x_min, y_min, width, height = bbox
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        x_center /= w
        y_center /= h
        width /= w
        height /= h

        line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))

def dhd_to_yolov5(
        shuffle: bool = True, 
        train_split: float = 0.6,
        val_split: float = 0.2, 
        test_split: float = 0.2,
        debug: bool = True
    ):
    
    # Define the content of the YAML file
    yaml_content = {
        'train': YOLO_IMG_TRAIN,
        'val': YOLO_IMG_VAL,
        'test': YOLO_IMG_TEST,
        'nc': 1,
        'names': ['pedestrian']
    }
    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    if not os.path.exists(yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"YAML file created at {YOLO_DIR}")
    else:
        print(f"YAML file already exists at {YOLO_DIR}")

    if (shuffle == True) and ((train_split + val_split + test_split) != 1.0):
        print("Train, val, test split must equal to 1.0")
        return 1
    try:
        filepath = ANN_TRAIN_DIR
        with open(filepath, 'r') as f:
            train_json = json.load(f)
        filepath = ANN_VAL_DIR
        with open(filepath, 'r') as f:
            val_json = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return 1
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {filepath}")
        return 1    
    if shuffle:
        total = len(train_json['images']) + len(val_json['images'])
        max_train  = math.ceil(total*train_split)
        max_val = math.ceil(total*val_split)
        max_test = math.ceil(total*test_split)
    train_id, val_id, test_id = [], [], []
    train, val, test = 0, 0, 0
    train_annot, val_annot = {}, {}

    for annot in train_json['annotations']:
        if annot['image_id'] not in train_annot:
            train_annot[annot['image_id']] = [annot['bbox']]
        else:
            train_annot[annot['image_id']].append(annot['bbox'])
    for annot in val_json['annotations']:
        if annot['image_id'] not in val_annot:
            val_annot[annot['image_id']] = [annot['bbox']]
        else:
            val_annot[annot['image_id']].append(annot['bbox'])

    for img in train_json['images']:
        choice = random.randint(1,3)
        chosen = False
        while not chosen:
            if choice == 1:
                if train == max_train:
                    choice = 2
                else:
                    train += 1
                    chosen = True
                    annot = train_annot[img['id']]
                    train_id.append({
                        'id': img['id'], 
                        'width': img['width'], 
                        'height': img['height'],
                        'annotations': train_annot[img['id']]
                    })
                    src_file = os.path.join(IMG_TRAIN_DIR, f"{img['id']}.jpg")
                    dst_dir = YOLO_IMG_TRAIN
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                    yolo_label(img['id'], train_annot[img['id']], img['width'], img['height'], 'train')
                    if debug:
                        print(f"Image ID {img['id']} has been added to train set. {train}/{max_train}")
            if choice == 2:
                if val == max_val:
                    choice = 3
                else:
                    val += 1
                    chosen = True
                    val_id.append({
                        'id': img['id'], 
                        'width': img['width'], 
                        'height': img['height'],
                        'annotations': train_annot[img['id']]
                    })
                    src_file = os.path.join(IMG_TRAIN_DIR, f"{img['id']}.jpg")
                    dst_dir = YOLO_IMG_VAL
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                    yolo_label(img['id'], train_annot[img['id']], img['width'], img['height'], 'val')
                    if debug:
                        print(f"Image ID {img['id']} has been added to val set. {val}/{max_val}")

            if choice == 3:
                if test == max_test:
                    choice = 1
                else:
                    test += 1
                    chosen = True
                    test_id.append({
                        'id': img['id'], 
                        'width': img['width'], 
                        'height': img['height'],
                        'annotations': train_annot[img['id']]
                    })
                    src_file = os.path.join(IMG_TRAIN_DIR, f"{img['id']}.jpg")
                    dst_dir = YOLO_IMG_TEST
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                    yolo_label(img['id'], train_annot[img['id']], img['width'], img['height'], 'test')
                    if debug:
                        print(f"Image ID {img['id']} has been added to test set. {test}/{max_test}")


    for img in val_json['images']:
        choice = random.randint(1,3)
        chosen = False
        while not chosen:
            if choice == 1:
                if train == max_train:
                    choice = 2
                else:
                    train += 1
                    chosen = True
                    train_id.append({
                        'id': img['id'], 
                        'width': img['width'], 
                        'height': img['height'],
                        'annotations': val_annot[img['id']]
                    })
                    src_file = os.path.join(IMG_VAL_DIR, f"{img['id']}.jpg")
                    dst_dir = YOLO_IMG_TRAIN
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                    yolo_label(img['id'], val_annot[img['id']], img['width'], img['height'], 'train')
                    if debug:
                        print(f"Image ID {img['id']} has been added to train set. {train}/{max_train}")
            if choice == 2:
                if val == max_val:
                    choice = 3
                else:
                    val += 1
                    chosen = True
                    val_id.append({
                        'id': img['id'], 
                        'width': img['width'], 
                        'height': img['height'],
                        'annotations': val_annot[img['id']]
                    })
                    src_file = os.path.join(IMG_VAL_DIR, f"{img['id']}.jpg")
                    dst_dir = YOLO_IMG_VAL
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                    yolo_label(img['id'], val_annot[img['id']], img['width'], img['height'], 'val')
                    if debug:
                        print(f"Image ID {img['id']} has been added to val set. {val}/{max_val}")

            if choice == 3:
                if test == max_test:
                    choice = 1
                else:
                    test += 1
                    chosen = True
                    test_id.append({
                        'id': img['id'], 
                        'width': img['width'], 
                        'height': img['height'],
                        'annotations': val_annot[img['id']]
                    })
                    src_file = os.path.join(IMG_VAL_DIR, f"{img['id']}.jpg")
                    dst_dir = YOLO_IMG_TEST
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                    yolo_label(img['id'], val_annot[img['id']], img['width'], img['height'], 'test')
                    if debug:
                        print(f"Image ID {img['id']} has been added to test set. {test}/{max_test}")


if __name__ == "__main__":
    dhd_to_yolov5()