import numpy as np
import cv2
import pandas as pd
import skimage.measure as skms
from scipy.stats import skew

def extract_color_moments(image, mask):
    """
    Extract mean, std, skewness for each channel in RGB and HSV spaces
    from the pixels inside the mask.
    """
    # Ensure boolean mask
    mask = mask.astype(bool)

    # Convert to both color spaces
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    features = []

    # RGB moments
    for c in range(3):
        pixels = rgb_image[:, :, c][mask]
        features.extend([
            pixels.mean(),
            pixels.std(ddof=0),
            skew(pixels)
        ])

    # HSV moments
    for c in range(3):
        pixels = hsv_image[:, :, c][mask]
        features.extend([
            pixels.mean(),
            pixels.std(ddof=0),
            skew(pixels)
        ])

    return np.array(features, dtype=float)

def extract_color_histograms(image, mask=None, bins=32):
    """Extract concatenated RGB and HSV histograms for a masked region."""
    if mask is not None:
        mask = mask.astype(bool)

    # Convert to RGB and HSV
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # RGB histograms
    rgb_hist = np.concatenate([
        np.histogram(rgb_image[:, :, 0][mask] if mask is not None else rgb_image[:, :, 0],
                     bins=bins, range=(0, 255))[0],
        np.histogram(rgb_image[:, :, 1][mask] if mask is not None else rgb_image[:, :, 1],
                     bins=bins, range=(0, 255))[0],
        np.histogram(rgb_image[:, :, 2][mask] if mask is not None else rgb_image[:, :, 2],
                     bins=bins, range=(0, 255))[0]
    ])

    # HSV histograms
    hsv_hist = np.concatenate([
        np.histogram(hsv_image[:, :, 0][mask] if mask is not None else hsv_image[:, :, 0],
                     bins=bins, range=(0, 179))[0],
        np.histogram(hsv_image[:, :, 1][mask] if mask is not None else hsv_image[:, :, 1],
                     bins=bins, range=(0, 255))[0],
        np.histogram(hsv_image[:, :, 2][mask] if mask is not None else hsv_image[:, :, 2],
                     bins=bins, range=(0, 255))[0]
    ])

    return np.concatenate([rgb_hist, hsv_hist]).ravel()