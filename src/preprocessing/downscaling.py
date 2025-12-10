import os
import sys
import tempfile  # 用於創建臨時文件
import subprocess  # 用於呼叫 DPID / SAID 等外部資源

import cv2  # OpenCV 影像處理 (BGR)
import numpy as np
import torch
import torch.nn.functional as F
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, os.pardir)
project_root = os.path.join(src_dir, os.pardir)

# DPID 可執行檔路徑
dpid_exe_path = os.path.join(project_root, "dpid", "dpid.exe")

# SAID 相關路徑
said_dir = os.path.join(project_root, "SAID")
said_pretrained_dir = os.path.join(said_dir, "pretrained_models")

_said_models_cache = {}
_said_models_module = None


def _load_said_models_module():
    """
    動態載入 SAID 專案中的 models.py，使其可在 src/ 中被使用，
    而不需要把 SAID 變成 Python package。
    """
    global _said_models_module
    if _said_models_module is not None:
        return _said_models_module
    # 確保 SAID 目錄在 sys.path 裡，讓 models.py 裡的 `from utils.utils` 可以找到 SAID/utils/utils.py
    if said_dir not in sys.path:
        sys.path.insert(0, said_dir)

    models_path = os.path.join(said_dir, "models.py")
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"SAID models.py not found at {models_path}")

    spec = importlib.util.spec_from_file_location("said_models", models_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _said_models_module = module
    return _said_models_module


def _get_said_model(model_name: str = "SAID_Bicubic", device: torch.device | None = None):
    """
    載入並快取 SAID 模型（SAID_Bicubic 或 SAID_Lanczos）。
    """
    global _said_models_cache

    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    key = (model_name, str(device))
    if key in _said_models_cache:
        return _said_models_cache[key]

    models_module = _load_said_models_module()
    model_dict = {
        "SAID_Bicubic": getattr(models_module, "SAID_Bicubic"),
        "SAID_Lanczos": getattr(models_module, "SAID_Lanczos"),
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown SAID model name: {model_name}")

    ckpt_path = os.path.join(said_pretrained_dir, model_name + ".pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAID checkpoint not found at {ckpt_path}")

    print(f"[INFO] Loading SAID model '{model_name}' from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "args" not in ckpt or "sd" not in ckpt:
        raise ValueError("SAID checkpoint format not expected. Missing 'args' or 'sd' keys.")

    ModelClass = model_dict[model_name]
    model = ModelClass(**ckpt["args"])
    model.load_state_dict(ckpt["sd"])
    model.to(device)
    model.eval()

    _said_models_cache[key] = model
    return model

def _downscale_lanczos_raw(image_input: (str, np.ndarray), target_size: tuple):
    """
    使用 Lanczos 插值將圖像下採樣到指定尺寸。

    Args:
        image_input (str, numpy.ndarray): 輸入圖像的路徑或 numpy 陣列。
        target_size (tuple): 目標尺寸，格式為 (寬度, 高度)。

    Returns:
        numpy.ndarray: 下採樣後的圖像 (OpenCV BGR 格式)。
    """
    # 處理輸入：優先檢查 numpy 陣列，然後是檔案路徑字串
    if isinstance(image_input, np.ndarray):
        img_np_bgr = image_input
    elif isinstance(image_input, str):
        img_np_bgr = cv2.imread(image_input)
        if img_np_bgr is None:
            raise FileNotFoundError(f"Image not found at {image_input}")
    else:
        raise TypeError("Unsupported image_input type. Expected str (path) or numpy.ndarray.")

    # 使用 OpenCV 的 resize 函數和 LANCZOS4 插值進行下採樣
    downscaled_img_np_bgr = cv2.resize(img_np_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return downscaled_img_np_bgr


# 新增圖像填充函數：先 padding 成正方形
def pad_to_square_and_downscale_friendly(
    image_np: np.ndarray,
    target_base_dim: int = 224,
    fill_color=(0, 0, 0),
) -> np.ndarray:
    """
    將圖像填充為正方形，並使其邊長為 target_base_dim 的倍數，以便於後續的迭代下採樣。

    Args:
        image_np (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。
        target_base_dim (int): 目標下採樣的最小邊長 (例如 224)。填充後的邊長將是此值的倍數。
        fill_color (tuple): 填充邊緣的顏色 (BGR 格式)。預設為黑色 (0, 0, 0)。

    Returns:
        numpy.ndarray: 填充後的正方形圖像。
    """
    height, width = image_np.shape[:2]
    # 這裡只確保圖像變成正方形，不再強制是 target_base_dim 的倍數
    padded_side = max(height, width)

    # 計算填充後的圖像應該放在新圖像的哪個位置 (置中)
    pad_h = (padded_side - height) // 2
    pad_w = (padded_side - width) // 2

    # 創建一個新的空白圖像，用指定顏色填充
    padded_image = np.full((padded_side, padded_side, 3), fill_color, dtype=np.uint8)

    # 將原始圖像複製到新圖像的中心
    padded_image[pad_h:pad_h + height, pad_w:pad_w + width] = image_np

    # print(f"Image padded from {width}x{height} to {padded_side}x{padded_side} for downscaling to {target_base_dim}x{target_base_dim}.")

    return padded_image

# 方案 1：只有 Lanczos，下採樣到 224x224
def Lanczos(image_input_np: np.ndarray) -> np.ndarray:
    """
    使用 Lanczos 插值將圖像填充為正方形並下採樣到 224x224。
    Args:
        image_input_np (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。
    Returns:
        numpy.ndarray: 處理後的圖像 (OpenCV BGR 格式)。
    """
    padded_img_np = pad_to_square_and_downscale_friendly(image_input_np, target_base_dim=224)
    return _downscale_lanczos_raw(padded_img_np, target_size=(224, 224))


def _downscale_dpid_raw(image_input: np.ndarray, target_size: tuple, lambda_val: float = 1.0) -> np.ndarray:
    """
    使用 DPID 演算法（透過 dpid.exe）將圖像下採樣到指定尺寸。

    Args:
        image_input (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。
        target_size (tuple): 目標尺寸，格式為 (寬度, 高度)。
        lambda_val (float): DPID 參數 lambda。

    Returns:
        numpy.ndarray: 下採樣後的圖像 (OpenCV BGR 格式)。
    """
    temp_input_path = None
    temp_output_path = None
    processed_img_np = None

    try:
        if not isinstance(image_input, np.ndarray):
            raise TypeError("Input image_input must be a numpy.ndarray.")

        if not os.path.exists(dpid_exe_path):
            raise FileNotFoundError(f"DPID executable not found at {dpid_exe_path}")

        # 建立暫存輸入檔案
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input_file:
            temp_input_path = temp_input_file.name

        cv2.imwrite(temp_input_path, image_input)

        work_dir = os.path.dirname(temp_input_path)
        base_name = os.path.basename(temp_input_path)

        out_w, out_h = target_size
        # 需與 Go 版本 main.go 的命名規則一致：strconv.FormatFloat(_lambda, 'f', -1, 32)
        # Python 對應行為可用 '%g'，例如 1.0 -> '1', 0.5 -> '0.5'
        lambda_str = ("%g" % float(lambda_val))

        # 呼叫 dpid.exe
        cmd = [
            dpid_exe_path,
            temp_input_path,
            str(out_w),
            str(out_h),
            lambda_str,
        ]

        subprocess.run(cmd, check=True, cwd=work_dir)

        # 根據 dpid.py 的命名規則組合輸出檔名：filename_wxh_lambda.png
        output_filename = f"{base_name}_{out_w}x{out_h}_{lambda_str}.png"
        temp_output_path = os.path.join(work_dir, output_filename)

        processed_img_np = cv2.imread(temp_output_path)
        if processed_img_np is None:
            raise IOError(f"Could not read processed image from {temp_output_path}")

    except Exception as e:
        print(f"Error during DPID downscaling: {e}")
        processed_img_np = image_input  # 發生錯誤時回傳原圖
    finally:
        # 清理暫存檔
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    return processed_img_np


def DPID(image_input_np: np.ndarray, lambda_val: float = 1.0) -> np.ndarray:
    """
    使用 DPID 演算法：先 padding 成正方形，再下採樣到 224x224。

    Args:
        image_input_np (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。
        lambda_val (float): DPID 參數 lambda。

    Returns:
        numpy.ndarray: 處理後的 224x224 圖像 (OpenCV BGR 格式)。
    """
    # 先填充成正方形，方便後續 224x224 下採樣
    padded_img_np = pad_to_square_and_downscale_friendly(image_input_np, target_base_dim=224)
    return _downscale_dpid_raw(padded_img_np, target_size=(224, 224), lambda_val=lambda_val)
def _downscale_said_from_rgb_square(
    image_rgb: np.ndarray,
    scale: float,
    target_size: int = 224,
    model_name: str = "SAID_Lanczos",
) -> np.ndarray:
    """
    使用 SAID 演算法，給定一張「已是正方形」的 RGB 影像，
    先按照固定倍率 scale 生出 LR，再 Bicubic resize 到 target_size x target_size。
    回傳值為 [0,1] 的 float32 RGB 影像。
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB image.")

    h, w, _ = image_rgb.shape
    if h != w:
        raise ValueError("image_rgb must be square for SAID processing.")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = _get_said_model(model_name=model_name, device=device)

    img = image_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # 1,C,H,W

    gt_size = [h, w]
    lr_size = [int(gt_size[0] / scale), int(gt_size[1] / scale)]

    with torch.no_grad():
        lr, _ = model(x, lr_size, gt_size)  # 1,C,H_lr,W_lr
        lr = lr.clamp(0.0, 1.0)
        lr_resized = F.interpolate(
            lr,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )

    lr_resized = lr_resized.cpu().squeeze(0).permute(1, 2, 0).numpy()  # H,W,C, 0~1
    return lr_resized


# 方案 2：先用 Lanczos 降到 448 或 896，再用 SAID(Lanczos) 降 2x / 4x 到 224x224。
# 為了讓 SAID 在邊界看到較自然的內容，我們對 SAID 使用「鏡射 padding」，
# 但在最終 224x224 輸出上，會把對應於 padding 的區域重新設為黑色。
def Lanczos_SAID(
    image_input_np: np.ndarray,
    said_model_name: str = "SAID_Lanczos",
) -> np.ndarray:
    """
    Pipeline:
    1. 以鏡射方式將輸入圖像 padding 成正方形（給 SAID 使用，邊界較自然）
    2. 如果正方形邊長 < 896：用 Lanczos 降到 448x448，再讓 SAID 以 scale=2 做 224x224
       否則：用 Lanczos 降到 896x896，再讓 SAID 以 scale=4 做 224x224
    3. 在最終 224x224 輸出上，把對應於「padding 區域」的像素設為黑色，保留原始內容區域

    回傳值：
        224x224 的 BGR 影像（uint8），中央為 SAID 輸出，外圈為黑色 padding。
    """
    # 原圖尺寸
    orig_h, orig_w = image_input_np.shape[:2]
    side = max(orig_h, orig_w)

    # 計算置中的 padding（上下左右）
    pad_top = (side - orig_h) // 2
    pad_bottom = side - orig_h - pad_top
    pad_left = (side - orig_w) // 2
    pad_right = side - orig_w - pad_left

    # 1) 以鏡射方式 padding 成正方形（給 SAID 使用）
    padded_bgr = cv2.copyMakeBorder(
        image_input_np,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    # 2) 決定中間尺寸與 SAID 的 scale
    if side < 896:
        intermediate_side = 448
        said_scale = 2.0
    else:
        intermediate_side = 896
        said_scale = 4.0

    # 3) 將 BGR 轉成 RGB，再用 Lanczos resize 到中間尺寸
    padded_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    inter_rgb = cv2.resize(
        padded_rgb,
        (intermediate_side, intermediate_side),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # 4) 丟給 SAID 做 scale=2 或 4 的 downscaling，再 Bicubic 到 224x224
    try:
        said_out_rgb = _downscale_said_from_rgb_square(
            inter_rgb,
            scale=said_scale,
            target_size=224,
            model_name=said_model_name,
        )
        # 5) 轉回 BGR + uint8
        said_out_rgb = (said_out_rgb * 255.0).clip(0, 255).astype(np.uint8)
        said_out_bgr = cv2.cvtColor(said_out_rgb, cv2.COLOR_RGB2BGR)

        # 6) 根據原始 padding 位置，在 224x224 上估計對應區域，將其設為黑色
        final_size = 224
        scale_to_final = final_size / float(side)

        y_start = int(round(pad_top * scale_to_final))
        y_end = int(round((pad_top + orig_h) * scale_to_final))
        x_start = int(round(pad_left * scale_to_final))
        x_end = int(round((pad_left + orig_w) * scale_to_final))

        # 邊界保護
        y_start = max(0, min(final_size, y_start))
        y_end = max(0, min(final_size, y_end))
        x_start = max(0, min(final_size, x_start))
        x_end = max(0, min(final_size, x_end))

        # 建立 mask：1 表示原始內容區域，0 表示 padding 區域
        mask = np.zeros((final_size, final_size), dtype=np.uint8)
        if y_end > y_start and x_end > x_start:
            mask[y_start:y_end, x_start:x_end] = 1

        # 將 padding 區域設為黑色（BGR = 0）
        said_out_bgr[mask == 0] = 0

        return said_out_bgr
    except Exception as e:
        # 若 SAID 失敗，退回純 Lanczos 結果，至少流程不會中斷
        print(f"[WARN] SAID downscaling failed ({e}), falling back to pure Lanczos.")
        return _downscale_lanczos_raw(padded_bgr, target_size=(224, 224))
