import importlib
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.option import args
from flask import Flask, request
import torch.nn.functional as F

ROOT_DIR = "/mnt/c/Adrianov/Projects/Project-Satanael"

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

def inpaint(impath, mpath, filename, args):
    # Model and version
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
    model.eval()

    out_dir = os.path.join(ROOT_DIR, 'results', 'inpainted')
    os.makedirs(out_dir, exist_ok=True)

    # iteration through datasets
    image = ToTensor()(Image.open(impath).convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(mpath).convert("L"))
    mask = mask.unsqueeze(0)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img = model(image_masked, mask)

    print("Image shape:", image.shape)       # expected (B, C, H, W)
    print("Mask shape:", mask.shape)         # could be (B, 1, H, W) or (H, W)
    print("Pred_img shape:", pred_img.shape) # expected (B, C, H, W)
    # Ensure image matches pred_img (B, C, H, W)
    # if image.shape[2:] != pred_img.shape[2:]:
    #     image = F.interpolate(image, size=pred_img.shape[2:], mode="bilinear", align_corners=False)

    # # Ensure mask matches pred_img (B, 1, H, W)
    # if mask.shape[2:] != pred_img.shape[2:]:
    #     mask = F.interpolate(mask.float(), size=pred_img.shape[2:], mode="nearest")

    # Make sure mask has same dims as image
    # if mask.shape[1] != image.shape[1]:
    #     mask = mask.repeat(1, image.shape[1], 1, 1)

    comp_imgs = (1 - mask) * image + mask * pred_img
    postprocess(image_masked[0]).save(os.path.join(out_dir, f"{filename}_masked.png"))
    postprocess(pred_img[0]).save(os.path.join(out_dir, f"{filename}_pred.png"))
    postprocess(comp_imgs[0]).save(os.path.join(out_dir, f"{filename}_comp.png"))
    print(f"saving to {os.path.join(out_dir, filename)}")

app = Flask(__name__)

@app.route("/inpaint", methods=["POST"])
def upload():
    temp_dir = os.path.join(ROOT_DIR, 'results', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    img = request.files["image"]; mask = request.files["mask"]
    img.save(os.path.join(temp_dir, img.filename))
    mask.save(os.path.join(temp_dir, mask.filename))
    inpaint(os.path.join(temp_dir, img.filename), os.path.join(temp_dir, mask.filename), mask.filename, args)
    return "Inference complete"

if __name__ == "__main__":
    app.run(port=1337)