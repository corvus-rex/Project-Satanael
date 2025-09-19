import os
from pathlib import Path
import shutil

patches = [
    'Naturalistic1', 'Naturalistic2', 'Naturalistic3',
    'Naturalistic4', 'Naturalistic5', 'Naturalistic6',
    'TSEA1', 'TSEA2'
]

input_imgs_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\images\merged")
input_labels_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\labels\merged")
copy_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\eval_final_patched\Test")
out_dir = Path(r"C:\Adrianov\Projects\Project-Satanael\data\tju-dhd\eval_final_patched\Clean_Train")

files = []
for patch in patches:
    # dir = os.path.join(copy_dir, patch, 'labels')
    # for l in os.listdir(dir):
    #     os.rename(os.path.join(dir, l), os.path.join(dir, f"{patch}_{l}"))

    os.makedirs(os.path.join(out_dir, patch, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, patch, 'labels'), exist_ok=True)
    prefix = patch+"_"
    dir = os.path.join(copy_dir, patch, 'images')
    files = [f[len(prefix):] for f in os.listdir(dir) if f.startswith(prefix)]
    for file in files:
        src_img = os.path.join(input_imgs_dir, file)
        src_label = os.path.join(input_labels_dir, file.replace(".jpg", ".txt"))
        dst_img_dir = os.path.join(out_dir, patch, 'images')
        dst_label_dir = os.path.join(out_dir, patch, 'labels')
        shutil.copy(src_img, dst_img_dir)
        shutil.copy(src_label, dst_label_dir)

