import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def recompress_diff(imorig, checkDisplacements):
    minQ = 51
    maxQ = 100
    stepQ = 1

    if checkDisplacements == 1:
        maxDisp = 7
    else:
        maxDisp = 0

    mins = []
    Output = []

    smoothing_b = 17
    Offset = (smoothing_b - 1) // 2

    # print(imorig.shape)
    height, width, _ = imorig.shape
    print("height , width", height, width)

    dispImages = []

    for ii in range(minQ, maxQ + 1, stepQ):
        cv2.imwrite('tmpResave.jpg', imorig, [int(cv2.IMWRITE_JPEG_QUALITY), ii])
        tmpResave = cv2.imread('tmpResave.jpg').astype(float)
        Deltas = []
        overallDelta = []

        for dispx in range(maxDisp + 1):
            for dispy in range(maxDisp + 1):
                DisplacementIndex = dispx * 8 + dispy + 1
                tmpResave_disp = tmpResave[dispx:, dispy:, :]
                imorig_disp = imorig[:height-dispx, :width-dispy, :].astype(float)
                # print('imorig_disp.shape', imorig_disp.shape)
                # print('tmpResave_disp.shape', tmpResave_disp.shape)
                Comparison = np.square(imorig_disp - tmpResave_disp)

                h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2
                Comparison = cv2.filter2D(Comparison, -1, h)

                Comparison = Comparison[Offset:-Offset, Offset:-Offset, :]
                Deltas.append(np.mean(Comparison, axis=2))
                overallDelta.append(np.mean(Deltas[DisplacementIndex - 1]))

        minOverallDelta, minInd = min(overallDelta), np.argmin(overallDelta)
        mins.append(minInd)
        Output.append(minOverallDelta)
        delta = Deltas[minInd]
        delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))

        dispImages.append(cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR))

    OutputY = Output
    OutputX = list(range(minQ, maxQ + 1, stepQ))
    xmax, imax, xmin, imin = cv2.minMaxLoc(np.array(OutputY))
    imin = sorted(imin)
    Qualities = [i * stepQ + minQ - 1 for i in imin]

    return OutputX, OutputY, dispImages, imin, Qualities, mins

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

def img_heatmap_cd(impath):
    im = clean_up_image(impath)
    checkDisplacements = 0
    # smoothFactor = 1
    OutputX, OutputY, dispImages, imin, Qualities, Mins = recompress_diff(im, checkDisplacements)
    OutputMap = dispImages

    return OutputMap, OutputX

