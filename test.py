import sys
import os
import importlib.util
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import warnings
from skimage import io, transform
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import threshold_local
import skimage.measure as skms
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import argparse
from tqdm import tqdm
import requests

compressdiff_path = os.path.abspath("./defenselib/spatial_heterogeneity.py")
spec = importlib.util.spec_from_file_location("compressdiff", compressdiff_path)
compressdiff = importlib.util.module_from_spec(spec)
sys.modules["compressdiff"] = compressdiff
spec.loader.exec_module(compressdiff)

color_path = os.path.abspath("./defenselib/feature_extraction/color.py")
spec = importlib.util.spec_from_file_location("color_ext", color_path)
color_ext = importlib.util.module_from_spec(spec)
sys.modules["color_ext"] = color_ext
spec.loader.exec_module(color_ext)

texture_path = os.path.abspath("./defenselib/feature_extraction/texture.py")
spec = importlib.util.spec_from_file_location("texture_ext", texture_path)
texture_ext = importlib.util.module_from_spec(spec)
sys.modules["texture_ext"] = texture_ext
spec.loader.exec_module(texture_ext)

def save_heatmap(img, path, cmap='grey'):
    plt.imsave(path, img, cmap=cmap)


def get_regions_from_mask(mask, min_area=60):
    label = skms.label(mask)
    props = skms.regionprops(label)
    regions = []
    for i, prop in enumerate(props):
        if prop.area >= min_area:
            region_mask = (label == i + 1).astype(np.uint8)
            regions.append({
                'mask': region_mask,
                'bbox': prop.bbox,
                'area': prop.area
            })
    return regions

def segment(impath, save_dir, kernel_pram=60):
    OutputMap, _ = compressdiff.img_heatmap_cd(impath)
    average_OutputMap = np.mean(OutputMap, axis=0)
    OutputMap_max = np.max(average_OutputMap)
    OutputMap_min = np.min(average_OutputMap)
    out_height = len(average_OutputMap)
    out_width = len(average_OutputMap[0])
    average_OutputMap = [int((average_OutputMap[i][j]-OutputMap_min)*255/(OutputMap_max-OutputMap_min)) for i in range(out_height) for j in range(out_width)]
    flatNumpyArray = np.array(average_OutputMap,dtype=np.uint8)

    # Convert the array to make a grayscale image
    grayImage = flatNumpyArray.reshape(out_height, out_width)
    img = cv2.imread(impath)
    ori_height, ori_width, _ = img.shape
    grayImage = cv2.resize(grayImage, (ori_width, ori_height)) 

    # Morphological processing
    base_kernel_size = int(min(ori_height, ori_width)/kernel_pram)
    kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
    opened = cv2.morphologyEx(grayImage, cv2.MORPH_OPEN,kernel, iterations=1)
    kernel=np.ones((base_kernel_size,base_kernel_size),np.uint8)
    closed=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,kernel, iterations=2)
    kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
    opened2=cv2.morphologyEx(closed,cv2.MORPH_OPEN,kernel, iterations=2)

    # Clamp all pixels below mean intensity
    non_zero_pixels = opened2[opened2 > 0]
    mean_intensity = np.mean(non_zero_pixels)
    filled = opened2.copy()
    filled[filled < mean_intensity] = mean_intensity
    filled = filled.astype(np.uint8)

    # Apply Otsu’s threshold
    _, thresh_map_adversarial = cv2.threshold(
        filled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    filename = Path(impath).stem

    # Save the heatmap
    save_heatmap(grayImage, os.path.join(save_dir, filename + "_cd.png"))
    save_heatmap(opened, os.path.join(save_dir, filename + "_cd_o.png"))
    save_heatmap(closed, os.path.join(save_dir, filename + "_cd_o_c.png"))
    save_heatmap(opened2, os.path.join(save_dir, filename + "_cd_o_c_o.png"))
    save_heatmap(filled, os.path.join(save_dir, filename + "_cd_filled.png"))
    save_heatmap(thresh_map_adversarial, os.path.join(save_dir, filename + "_cd_thresh.png"))

    print(f"Saved adversarial heatmap to {os.path.join(save_dir, filename + "_cd_thresh.png")}.")
    return thresh_map_adversarial

def extract_feature(image, region):
    BINS = 32
    DISTS = [1,2,4,8,16,32,64]
    ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    feats = [region['area']]
    feats.extend(color_ext.extract_color_histograms(image, region['mask'], BINS))
    feats.extend(color_ext.extract_color_moments(image, region['mask']))
    haralick = texture_ext.extract_haralick_features(image, region['mask'], DISTS, ANGLES)
    feats.extend(list(haralick.values()))
    return feats
    
def classify(impath, 
             mask_path, 
             scaler_path, 
             out_dir,
             xgb_path=None, 
             rf_path=None,
             models=['xgb'], 
             resize=1024):
    image = cv2.imread(impath)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    os.makedirs(out_dir, exist_ok=True)
    h,w = image.shape[:2]
    if resize > 0 or resize != None:
        if h < w:
            new_h = resize
            new_w = int(w * (resize / h))
        else:
            new_w = resize
            new_h = int(h * (resize / w))
        
        # --- Resize image ---
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    mask = (mask > 0).astype(np.uint8)

    regions = get_regions_from_mask(mask)
    scaler = joblib.load(scaler_path)

    if 'xgb' in models:
        xgb_loaded = xgb.XGBClassifier()
        xgb_loaded.load_model(xgb_path)
    if 'rf' in models:
        rf_loaded = joblib.load(rf_path)

    out = {}
    for m in models:
        out[m] = np.zeros((new_h, new_w), dtype=np.uint8)

    for _, r in enumerate(regions):
        # print("AAAA", image.shape)
        # print("BBBB", r['mask'].shape)
        feats = extract_feature(image, r)
        feats = np.array([feats])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            feats_norm = scaler.transform(feats)
        preds = {}
        if 'rf' in models:
            preds['rf'] = rf_loaded.predict(feats_norm)
        if 'xgb' in models:
            preds['xgb'] = xgb_loaded.predict(feats_norm)

        for m in preds:
            if preds[m] == 1:
                # print("CCCCC", np.unique(out[m]))
                # print("DDDDD", np.unique(r['mask']))
                out[m] = np.bitwise_or(out[m].astype(np.uint8), r['mask'])
                # print("O", np.unique(out[m]))

    filename = Path(impath).stem
    for m in models:
        path = os.path.join(out_dir, f"{filename}_mask_{m}.png")
        Image.fromarray(out[m] * 255).save(path)
        print(f"Adversarial mask saved to {path}")

if __name__ == "__main__":
    ROOT_DIR = "C:\\Adrianov\\Projects\\Project-Satanael\\"

    DIRS = ['Test']

    PATCH_TYPES = [
        'Naturalistic1', 'Naturalistic2', 'Naturalistic3',
        'Naturalistic4', 'Naturalistic5', 'Naturalistic6',
        'TSEA1', 'TSEA2'
    ]

    TEST_TYPES = ['Naturalistic5', 'Naturalistic6', 'TSEA2']

    RESIZE = 1024

    parser = argparse.ArgumentParser()
    parser.add_argument("--classify-only", action="store_true", dest="classify", help="Only performs feature extraction and classification")
    parser.add_argument("--inpaint-only", action="store_true", dest="inpaint", help="Only performs feature extraction and classification")
    args = parser.parse_args()
    print(args.classify)
    if args.classify:
        data_dir = os.path.join(ROOT_DIR, 'data', 'tju-dhd', 'eval_final_patched')
        mask_dir = os.path.join(ROOT_DIR, 'results_cd_grey_test_final_eval')
        out_dir = os.path.join(ROOT_DIR, 'results', 'adv_mask')
        models_dir = os.path.join(ROOT_DIR, 'models')
        scaler_path = os.path.join(models_dir, 'minmax_scaler.pkl')
        xgb_path = os.path.join(models_dir, 'xgboost_model.json')
        rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
        for d in DIRS:
            for patch in PATCH_TYPES:
                if d == 'Train' and patch in TEST_TYPES:
                    continue
                if d == 'Train':
                    label_dir = 'labels'
                if d == 'Test':
                    label_dir = 'labels_w_adv'

                img_dir = os.path.join(data_dir, d, patch, 'images')

                for fname in tqdm(os.listdir(img_dir), desc=f"Classifying adversarial regions {patch} {d}..."):
                    if not fname.endswith('.jpg'):
                        continue
                    impath = os.path.join(img_dir, fname)
                    mask_path = os.path.join(mask_dir, f"{fname}_cd_thresh.png")
                    classify(impath=impath, 
                            mask_path=mask_path, 
                            scaler_path=scaler_path,
                            out_dir=out_dir,
                            xgb_path=xgb_path,
                            rf_path=rf_path,
                            models=['xgb', 'rf'])
                    
    if args.inpaint:
        data_dir = os.path.join(ROOT_DIR, 'data', 'tju-dhd', 'eval_final_patched')
        mask_dir = os.path.join(ROOT_DIR, 'results', 'adv_mask')
        start = time.time()
        i = 0
        for d in DIRS:
            for patch in PATCH_TYPES:
                if d == 'Train' and patch in TEST_TYPES:
                    continue
                if d == 'Train':
                    label_dir = 'labels'
                if d == 'Test':
                    label_dir = 'labels_w_adv'

                img_dir = os.path.join(data_dir, d, patch, 'images')
                for fname in tqdm(os.listdir(img_dir), desc=f"Inpainting adversarial regions {patch} {d}..."):
                    fname_stemmed, _ = os.path.splitext(fname)
                    
                    # --- Load & resize image to 512x512 ---
                    image = cv2.imread(os.path.join(img_dir, fname))
                    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
                    _, img_bytes = cv2.imencode(".jpg", image)

                    # --- Load & resize mask to 512x512 ---
                    mask_path = os.path.join(mask_dir, f"{fname_stemmed}_mask_xgb.png")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # ensure single channel
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  # keep binary values
                    _, mask_bytes = cv2.imencode(".png", mask)

                    # --- Send request ---
                    files = {
                        "image": (fname, img_bytes.tobytes(), "image/jpeg"),
                        "mask": (os.path.basename(mask_path), mask_bytes.tobytes(), "image/png")
                    }
                    r = requests.post("http://127.0.0.1:1337/inpaint", files=files)
                    i += 1
        elapsed = time.time() - start
        print(f"{i} images processed. {elapsed:.2f} seconds elasped. Avg time: {elapsed/i:.2f} seconds")