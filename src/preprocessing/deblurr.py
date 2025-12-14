import cv2
import numpy as np
from scipy.signal import wiener

def ssk(image):
    """
    Algorithm 1: Simple Sharpening Kernel
    """

    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    
    enhanced_image = cv2.filter2D(image, -1, kernel)
    
    return enhanced_image

def usm(image, kernel_size=(15, 15), sigma=1.0, amount=5.0):

    """
    Algorithm 2: Unsharp Masking (USM)
    Enhanced = Original + Amount * (Original - Blurred)
    
    """

    # 1. 
    try:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    except Exception as e:
        print("f")
        return image  
    # 2. USM
    # src1 * alpha + src2 * beta + gamma
    # image * (1 + amount) + blurred * (-amount)
    enhanced_image = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return enhanced_image

def swf(image, kernel_size=(50, 50)):
    """
    Algorithm 3: Simplified Wiener Filter (Robust Implementation)
    """
    def apply_wiener(img_channel, k_size):
        # 1. float64
        img = img_channel.astype(np.float64)
        
        # 2. Local Mean -> mu
        mu = cv2.blur(img, k_size)
        
        # 3. mean(x^2)
        img_sq = img * img
        mu_sq = cv2.blur(img_sq, k_size)
        
        # 4. Local Variance -> sigma^2 = E[x^2] - (E[x])^2
        sigma_sq = mu_sq - (mu * mu)
        

        sigma_sq = np.maximum(sigma_sq, 0)
        
        # 5. Noise Variance -> nu^2

        noise_sq = np.mean(sigma_sq)
        
        # 6. Wiener: y = mu + (sigma^2 - noise^2) / sigma^2 * (x - mu)
        weight = (sigma_sq - noise_sq) / (sigma_sq + 1e-10)
        
        weight = np.maximum(weight, 0)
        
        res = mu + weight * (img - mu)
        
        return res

  
    if len(image.shape) == 3:
        channels = cv2.split(image)
        cleaned_channels = []
        for ch in channels:
            res = apply_wiener(ch, kernel_size)
            res = np.clip(res, 0, 255).astype(np.uint8)
            cleaned_channels.append(res)
        return cv2.merge(cleaned_channels)
    else:
        res = apply_wiener(image, kernel_size)
        return np.clip(res, 0, 255).astype(np.uint8)
    