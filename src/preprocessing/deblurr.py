import cv2
import numpy as np
from scipy.signal import wiener

def ssk(image):
    """
    Algorithm 1: Simple Sharpening Kernel
    利用固定的 High-pass filter 增強邊緣。
    """

    # 定義銳化核 (3x3 Laplacian variant)
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    
    # 使用 cv2.filter2D 進行卷積
    # ddepth=-1 表示保持原圖像深度 (uint8)
    enhanced_image = cv2.filter2D(image, -1, kernel)
    
    return enhanced_image

def usm(image, kernel_size=(15, 15), sigma=1.0, amount=5.0):

    """
    Algorithm 2: Unsharp Masking (USM)
    公式: Enhanced = Original + Amount * (Original - Blurred)
    
    [Ablation Study 可調參數]:
    - amount: 銳化強度 (建議範圍 0.5 ~ 2.0)
    - kernel_size: 模糊核大小 (建議 (3,3), (5,5), (9,9))
    """

    # 1. 產生模糊版本
    try:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    except Exception as e:
        print("f")
        return image  # 若模糊失敗，回傳原圖
    # 2. 計算 USM (使用 addWeighted 高效疊加)
    # src1 * alpha + src2 * beta + gamma
    # image * (1 + amount) + blurred * (-amount)
    enhanced_image = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return enhanced_image

def swf(image, kernel_size=(50, 50)):
    """
    Algorithm 3: Simplified Wiener Filter (Robust Implementation)
    手動實作版本，解決 scipy 除以零的警告問題。
    """
    def apply_wiener(img_channel, k_size):
        # 1. 轉換為 float64
        img = img_channel.astype(np.float64)
        
        # 2. 計算局部平均值 (Local Mean) -> mu
        mu = cv2.blur(img, k_size)
        
        # 3. 計算局部平方的平均值 -> mean(x^2)
        img_sq = img * img
        mu_sq = cv2.blur(img_sq, k_size)
        
        # 4. 計算局部變異數 (Local Variance) -> sigma^2 = E[x^2] - (E[x])^2
        sigma_sq = mu_sq - (mu * mu)
        
        # 防止變異數小於 0 (計算誤差)
        sigma_sq = np.maximum(sigma_sq, 0)
        
        # 5. 估計雜訊變異數 (Noise Variance) -> nu^2
        # 這裡我們假設雜訊變異數是所有局部變異數的平均值
        noise_sq = np.mean(sigma_sq)
        
        # 6. 套用 Wiener 公式: y = mu + (sigma^2 - noise^2) / sigma^2 * (x - mu)
        # 為了防止除以零，我們在分母加一個極小值 (1e-10)
        weight = (sigma_sq - noise_sq) / (sigma_sq + 1e-10)
        
        # 權重不能小於 0 (當局部變異數 < 雜訊時，視為純雜訊，權重設為 0 -> 完全平滑)
        weight = np.maximum(weight, 0)
        
        res = mu + weight * (img - mu)
        
        return res

    # 處理彩色或灰階
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
    