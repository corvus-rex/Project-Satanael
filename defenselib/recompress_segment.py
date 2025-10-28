import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def recompress_diff(imorig, smoothing_b=17, minQ=51, maxQ=100):

    mins = []
    Output = []

    offset = (smoothing_b - 1) // 2

    height, width, _ = imorig.shape
    disp_images = []

    for q in range(minQ, maxQ + 1):
        cv2.imwrite('jpg_recompress.jpg', imorig, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        jpg_resave = cv2.imread('jpg_recompress.jpg').astype(float)
        deltas = []
        overall_delta = []

        imorig_disp = imorig[:height, :width, :].astype(float)
        comparison = np.square(imorig_disp - jpg_resave)

        h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2
        comparison = cv2.filter2D(comparison, -1, h)

        comparison = comparison[offset:-offset, offset:-offset, :]
        deltas.append(np.mean(comparison, axis=2))
        overall_delta.append(np.mean(deltas[-1]))

        minOverallDelta, minInd = min(overall_delta), np.argmin(overall_delta)
        mins.append(minInd)
        Output.append(minOverallDelta)
        delta = deltas[minInd]
        delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta) + 5e-12)

        disp_images.append(cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR))

    return disp_images

def clean_up_image(filename):
    im = cv2.imread(filename)

    if len(im.shape) > 3:
        im = im[:, :, :, 0, 0, 0, 0]

    dots = filename.rfind('.')
    extension = filename[dots:]
    
    if extension.lower() == '.gif' and im.shape[2] < 3:
        im_gif, gif_map = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im_gif = im_gif[:, :, 0]
        im = np.uint8(cv2.cvtColor(im_gif, cv2.COLOR_GRAY2RGB) * 255)

    if im.shape[2] < 3:
        im[:, :, 1] = im[:, :, 0]
        im[:, :, 2] = im[:, :, 0]

    if im.shape[2] > 3:
        im = im[:, :, 0:3]

    if im.dtype == np.uint16:
        im = np.uint8(np.floor(im / 256))

    im_out = im

    return im_out

def heatmap_recompressdiff(impath):
    im = clean_up_image(impath)
    dispImages = recompress_diff(im)
    return dispImages

