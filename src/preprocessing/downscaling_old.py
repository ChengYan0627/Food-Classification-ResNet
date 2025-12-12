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





def _resize_short_side_lanczos(image_np: np.ndarray, target_short_side: int) -> np.ndarray:

    """

    使用 Lanczos 插值，將輸入影像的「短邊」縮放到 target_short_side，維持長寬比，不做 padding。



    Args:

        image_np (numpy.ndarray): 輸入圖像 (BGR)。

        target_short_side (int): 短邊目標長度，例如 256 或 512。



    Returns:

        numpy.ndarray: 縮放後的 BGR 圖像。

    """

    h, w = image_np.shape[:2]

    if h == 0 or w == 0:

        return image_np



    if h <= w:

        new_h = target_short_side

        new_w = int(round(w * target_short_side / float(h)))

    else:

        new_w = target_short_side

        new_h = int(round(h * target_short_side / float(w)))



    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return resized





def _center_crop(image_np: np.ndarray, crop_size: int = 224) -> np.ndarray:

    """

    從中心裁切成 crop_size x crop_size，不做 padding。

    假設輸入長寬皆 >= crop_size。

    """

    h, w = image_np.shape[:2]

    if h < crop_size or w < crop_size:

        # 若真的太小，就直接回傳原圖，避免崩潰（資料應該不會這樣）

        return image_np



    y0 = (h - crop_size) // 2

    x0 = (w - crop_size) // 2

    return image_np[y0:y0 + crop_size, x0:x0 + crop_size]





# 方案 1：只有 Lanczos：短邊縮到 256，再中心裁切 224x224

def Lanczos(image_input_np: np.ndarray) -> np.ndarray:

    """

    使用 Lanczos 插值做「幾何前處理」：

    1. 將輸入影像短邊縮放到 256（維持長寬比，不做 padding）

    2. 從縮放後影像中心裁切成 224x224



    這樣輸出的 224x224 影像即可直接餵給 classifier（搭配 processor.do_resize=False）。

    """

    resized = _resize_short_side_lanczos(image_input_np, target_short_side=256)

    cropped = _center_crop(resized, crop_size=224)

    return cropped





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

    使用 DPID 演算法做「幾何前處理」：

    1. 利用 DPID 將輸入影像的短邊縮放到 256（維持長寬比，不做 padding）

    2. 從縮放後影像中心裁切成 224x224



    Args:

        image_input_np (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。

        lambda_val (float): DPID 參數 lambda。



    Returns:

        numpy.ndarray: 處理後的 224x224 圖像 (OpenCV BGR 格式)。

    """

    h, w = image_input_np.shape[:2]

    if h == 0 or w == 0:

        return image_input_np



    # 計算目標尺寸：短邊 = 256，維持長寬比

    target_short = 256

    if h <= w:

        out_h = target_short

        out_w = int(round(w * target_short / float(h)))

    else:

        out_w = target_short

        out_h = int(round(h * target_short / float(w)))



    resized = _downscale_dpid_raw(image_input_np, target_size=(out_w, out_h), lambda_val=lambda_val)

    cropped = _center_crop(resized, crop_size=224)

    return cropped

def _downscale_said_from_rgb_square(

    image_rgb: np.ndarray,

    scale: float,

    target_size: int = 224,

    model_name: str = "SAID_Lanczos",

) -> np.ndarray:

    """

    使用 SAID 演算法，給定一張 RGB 影像，按照倍率 scale 產生 LR 影像。



    與原本版本不同之處在於：

    - 不再強制輸入為正方形，也不在這裡做額外的 Bicubic resize / crop

    - 輸出為 SAID 下採樣後的 LR，大小約為 (H/scale, W/scale)，uint8、值域在 [0,255]



    Args:

        image_rgb (numpy.ndarray): 輸入 RGB 影像，H x W x 3，值域 [0,255] 或 [0,1] 皆可。

            若為 [0,1] 會在此自動乘上 255，與原始 SAID 專案一致使用 0~255 範圍。

        scale (float): 目標下採樣倍率（例如 2.0）。

        target_size (int): 僅為向後相容保留，現在不再使用。

        model_name (str): 使用的 SAID 模型名稱。

    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:

        raise ValueError("image_rgb must be an HxWx3 RGB image.")



    h, w, _ = image_rgb.shape



    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = _get_said_model(model_name=model_name, device=device)



    # SAID 原始實作假設輸入範圍為 0~255 的 float tensor。

    # 這裡同時相容 0~255（uint8 或 float）與 0~1 的輸入：

    # - 若最大值 <= 1.5，視為 0~1，則乘上 255

    # - 否則視為已在 0~255 範圍

    img = image_rgb.astype(np.float32)

    if img.max() <= 1.5:

        img *= 255.0



    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)  # 1,C,H,W



    gt_size = [h, w]

    lr_size = [int(gt_size[0] / scale), int(gt_size[1] / scale)]



    # 使用 inference_mode + AMP（在 CUDA 上啟用混合精度）加速推論

    amp_enabled = device.type == "cuda"

    with torch.inference_mode():

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):

            lr, _ = model(x, lr_size, gt_size)  # 1,C,H_lr,W_lr，理論上範圍約在 0~255

            lr = lr.clamp(0.0, 255.0)



    # 轉為 H_lr x W_lr x 3、uint8（0~255），方便後續直接當作一般影像處理

    lr_np = lr.cpu().squeeze(0).permute(1, 2, 0).numpy()

    lr_np_uint8 = lr_np.astype(np.uint8)

    return lr_np_uint8





def Lanczos_SAID(

    image_input_np: np.ndarray,

    said_model_name: str = "SAID_Lanczos",

) -> np.ndarray:

    """

    方案 2：Lanczos + SAID，下採樣到 224x224，且不做 padding。



    Pipeline:

    1. 使用 Lanczos 將輸入影像短邊縮放到 512（維持長寬比，不做 padding）

    2. 將縮放後影像轉成 RGB，送入 SAID，以 scale=2 做下採樣，得到短邊約 256 的 LR 影像

    3. 對 SAID 輸出做中心裁切，得到 224x224



    回傳值：

        224x224 的 BGR 影像（uint8），可直接餵給 classifier（搭配 processor.do_resize=False）。

    """

    # 1) 先用 Lanczos 將短邊縮到 512

    resized_512_bgr = _resize_short_side_lanczos(image_input_np, target_short_side=512)



    # 2) BGR -> RGB，丟給 SAID 以 scale=2 下採樣

    resized_512_rgb = cv2.cvtColor(resized_512_bgr, cv2.COLOR_BGR2RGB)

    said_out_rgb = _downscale_said_from_rgb_square(

        resized_512_rgb,

        scale=2.0,

        target_size=256,  # 僅作為紀錄，實際未使用

        model_name=said_model_name,

    )



    # SAID wrapper 已回傳 uint8 RGB（0~255），此處僅確保型別正確

    said_out_rgb_uint8 = said_out_rgb.astype(np.uint8)



    # 3) 中心裁切成 224x224（仍在 RGB 空間）

    said_out_rgb_cropped = _center_crop(said_out_rgb_uint8, crop_size=224)



    # 轉回 BGR 作為 OpenCV / 其他處理的慣用格式

    said_out_bgr_cropped = cv2.cvtColor(said_out_rgb_cropped, cv2.COLOR_RGB2BGR)

    return said_out_bgr_cropped

