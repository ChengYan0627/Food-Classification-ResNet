import numpy as np
import cv2

def gamma(image, gamma=1.5):
    """
    Apply gamma correction to enhance low-light images.
    """

    inv_gamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma) * 255
    table = table.astype("uint8")

    # Apply lookup table
    return cv2.LUT(image, table)