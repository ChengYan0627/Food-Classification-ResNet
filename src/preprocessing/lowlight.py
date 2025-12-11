import numpy as np
import cv2

def low_light_check(image, threshold=60):
    """
    Check if an image is considered low-light based on average brightness.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold

def gamma(image, gamma=1.5):
    """
    Apply gamma correction to enhance low-light images.
    """

    if low_light_check(image) == False:
        return image

    inv_gamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma) * 255
    table = table.astype("uint8")

    # Apply lookup table
    return cv2.LUT(image, table)

def CLAHE(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance low-light images.
    
    Source: https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
    """

    if low_light_check(image) == False:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

def SSRetinex(image, sigma=30):
    """
    Apply Single Scale Retinex to enhance low-light images.
    
    Source: https://github.com/AKRISH22/Retinex-Image-Enhancement
    """

    if low_light_check(image) == False:
        return image

    img = image.astype(np.float32) + 1.0

    # SSR formula
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log(img) - np.log(blur)

    # Normalize
    retinex_min = np.min(retinex)
    retinex_max = np.max(retinex)
    retinex_norm = (retinex - retinex_min) * (255.0 / (retinex_max - retinex_min))

    return retinex_norm.astype(np.uint8)