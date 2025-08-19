import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import gabor
from skimage.util import img_as_float

def extract_haralick_features(image, mask, 
                                   distances=[1, 2, 4, 8], 
                                   angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                   levels=256):
    """
    Extract Haralick features from the masked region of an image,
    returned as a dictionary (one key per distance+angle combination).
    
    Parameters:
        image (ndarray): Grayscale input image.
        mask (ndarray): Binary mask (same shape as image). Non-zero = region of interest.
        distances (list): Pixel pair distance offsets.
        angles (list): Pixel pair angles in radians.
        levels (int): Number of gray levels for GLCM.
    
    Returns:
        features (dict): Dictionary of Haralick features.
                         Keys like 'contrast_dist8_90deg'.
    """
    # Ensure uint8 for GLCM
    image = img_as_ubyte(image)

    # Apply mask (set outside region to 0)
    gray = rgb2gray(image)          # shape (1820, 1024)
    gray = img_as_ubyte(gray)       # convert to uint8

    masked_image = np.where(mask > 0, gray, 0)
    # Compute GLCM
    glcm = graycomatrix(masked_image, 
                        distances=distances, 
                        angles=angles, 
                        levels=levels, 
                        symmetric=True, 
                        normed=True)

    # Define Haralick properties
    props = ['contrast', 'dissimilarity', 'homogeneity', 
             'energy', 'correlation', 'ASM']

    # Convert angles (radians → degrees for readable keys)
    angle_degrees = {a: int(np.round(np.degrees(a))) for a in angles}

    features = {}

    # Extract features
    for prop in props:
        values = graycoprops(glcm, prop)  # shape = (len(distances), len(angles))
        for d_idx, d in enumerate(distances):
            for a_idx, a in enumerate(angles):
                key = f"{prop}_dist{d}_{angle_degrees[a]}deg"
                features[key] = values[d_idx, a_idx]

    return features

def extract_gabor_features(image, mask, 
                           frequencies=[0.1, 0.2, 0.3, 0.4], 
                           thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract Gabor texture features from the masked region of an image.
    
    Parameters:
        image (ndarray): Input image (RGB or grayscale).
        mask (ndarray): Binary mask (same HxW as image). Non-zero = region of interest.
        frequencies (list): Gabor filter frequencies (cycles per pixel).
        thetas (list): Gabor filter orientations (radians).
    
    Returns:
        features (dict): Dictionary of extracted features.
                         Keys like 'gabor_mean_freq0.2_theta45'.
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_float(image)

    # Apply mask
    masked_image = np.where(mask > 0, image, 0)

    features = {}
    for freq in frequencies:
        for theta in thetas:
            # Apply Gabor filter
            real, imag = gabor(masked_image, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)

            # Extract statistics only inside mask
            values = magnitude[mask > 0]
            if values.size > 0:  # avoid empty ROI
                mean_val = values.mean()
                var_val = values.var()
                energy_val = np.sum(values**2) / values.size
                std_val = values.std()

                angle_deg = int(np.round(np.degrees(theta)))
                features[f"gabormean_freq{freq:.2f}_{angle_deg}deg"] = mean_val
                features[f"gaborvar_freq{freq:.2f}_{angle_deg}deg"] = var_val
                features[f"gaborenergy_freq{freq:.2f}_{angle_deg}deg"] = energy_val
                features[f"gaborstd_freq{freq:.2f}_{angle_deg}deg"] = std_val
            else:
                # Fill with NaN if mask is empty
                angle_deg = int(np.round(np.degrees(theta)))
                features[f"gabormean_freq{freq:.2f}_{angle_deg}deg"] = np.nan
                features[f"gaborvar_freq{freq:.2f}_{angle_deg}deg"] = np.nan
                features[f"gaborenergy_freq{freq:.2f}_{angle_deg}deg"] = np.nan
                features[f"gaborstd_freq{freq:.2f}_{angle_deg}deg"] = np.nan

    return features