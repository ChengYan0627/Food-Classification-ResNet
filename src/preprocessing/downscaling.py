import os
import sys
import torch
import numpy as np
import importlib.util
import cv2 # Added for cv2.cvtColor
from transformers import AutoImageProcessor # 新增導入 AutoImageProcessor
import matlab.engine # 導入 matlab.engine
import tempfile # 用於創建臨時文件
import math # 新增導入 math 模組

# AIDN 儲存庫的路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, os.pardir)
project_root = os.path.join(src_dir, os.pardir)
aidn_repo_path = os.path.join(project_root, 'AIDN_repo')

# 動態載入 AIDN 相關模組

def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module # 將模組添加到 sys.modules，以便內部導入能找到它
    spec.loader.exec_module(module)
    return module

# 載入 BaseModel (base.base_model) for AIDN
base_model_path = os.path.join(aidn_repo_path, 'base', 'base_model.py')
base_module = _load_module_from_path("base.base_model", base_model_path)
BaseModel = base_module.BaseModel

# 載入 models.common for AIDN
common_path = os.path.join(aidn_repo_path, 'models', 'common.py')
common_module = _load_module_from_path("models.common", common_path)

# 載入 models.arb for AIDN
arb_path = os.path.join(aidn_repo_path, 'models', 'arb.py')
arb_module = _load_module_from_path("models.arb", arb_path)

# 載入 models.lib.quantization for AIDN
quantization_path = os.path.join(aidn_repo_path, 'models', 'lib', 'quantization.py')
quantization_module = _load_module_from_path("models.lib.quantization", quantization_path)
Quantization = quantization_module.Quantization
Quantization_RS = quantization_module.Quantization_RS

# 載入 models.arbedrs (依賴於 models.common 和 models.arb) for AIDN
arbedrs_path = os.path.join(aidn_repo_path, 'models', 'arbedrs.py')
arbedrs_module = _load_module_from_path("models.arbedrs", arbedrs_path)
EDRS = arbedrs_module.EDRS

# 載入 models.inv_arb_edrs (依賴於 base.base_model, models.arbedrs, models.lib.quantization) for AIDN
inv_arb_edrs_path = os.path.join(aidn_repo_path, 'models', 'inv_arb_edrs.py')
inv_arb_edrs_module = _load_module_from_path("models.inv_arb_edrs", inv_arb_edrs_path)
InvArbEDRS = inv_arb_edrs_module.InvArbEDRS

class AIDNConfig:
    # 根據 AIDN_repo/config/DIV2K/AIDN.yaml 創建一個簡化的配置物件
    def __init__(self):
        self.arch = 'InvEDRS_arb'
        self.up_sampler = 'sampleB'
        self.down_sampler = 'sampleB'
        self.n_resblocks = 16
        self.n_feats = 64
        self.fixed_scale = False
        self.n_colors = 3
        self.res_scale = 1
        self.quantization = True
        self.quantization_type = 'round_soft'
        self.K = 4
        self.num_experts_SAconv = 4
        self.num_experts_CRM = 8
        self.jpeg = False
        self.rgb_range = 1.0
        self.rescale = 'down' # 確保模型在初始化時知道是要下採樣

# 載入 AIDN 模型和權重的實例
_aidn_model_instance = None
_aidn_model_weights_path = None

def _get_aidn_model(model_weights_path: str):
    global _aidn_model_instance, _aidn_model_weights_path

    if _aidn_model_instance is None or _aidn_model_weights_path != model_weights_path:
        # print(f"Initializing AIDN model with weights from {model_weights_path}") # Removed debug print
        device = torch.device('cpu') # Changed from 'cuda' if available
        cfg = AIDNConfig()
        model = InvArbEDRS(cfg).to(device)

        if model_weights_path:
            # 載入模型權重
            state_dict = torch.load(model_weights_path, map_location=device)['state_dict']

            # 移除 state_dict 中的 'module.' 前綴
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
            # print("AIDN model weights loaded successfully.") # Removed debug print

        model.eval()
        _aidn_model_instance = model
        _aidn_model_weights_path = model_weights_path
    return _aidn_model_instance

def _preprocess_image(image_input):
    # 處理輸入：優先檢查 numpy 陣列，然後是檔案路徑字串
    if isinstance(image_input, np.ndarray):
        # OpenCV 讀取的圖像通常是 BGR 格式，PIL 期望 RGB
        img_np_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        img_size = (image_input.shape[1], image_input.shape[0]) # (width, height)
    elif isinstance(image_input, str):
        img_np_bgr = cv2.imread(image_input)
        if img_np_bgr is None:
            raise FileNotFoundError(f"Image not found at {image_input}")
        img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
        img_size = (img_np_bgr.shape[1], img_np_bgr.shape[0]) # (width, height)
    else:
        raise TypeError("Unsupported image_input type. Expected str (path) or numpy.ndarray.")

    img_np = img_np_rgb.astype(np.float32) / 255.0 # 歸一化到 [0, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(torch.device('cpu'))
    return img_tensor, img_size

def _postprocess_image(img_tensor):
    # 從 PyTorch Tensor 轉換回 OpenCV (BGR) NumPy 陣列
    img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img_np_rgb = (img_tensor * 255.0).astype(np.uint8)
    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR) # 轉換為 BGR 格式給 OpenCV
    return img_np_bgr

def downscale_with_aidn(image_input: (str, np.ndarray), target_size: tuple, model_weights_path: str):
    """
    使用 AIDN 模型將圖像下採樣到指定尺寸。

    Args:
        image_input (str, numpy.ndarray): 輸入圖像的路徑或 numpy 陣列。
        target_size (tuple): 目標尺寸，格式為 (寬度, 高度)。
        model_weights_path (str): AIDN 模型權重的路徑。

    Returns:
        numpy.ndarray: 下採樣後的圖像 (OpenCV BGR 格式)。
    """
    model = _get_aidn_model(model_weights_path)
    
    input_tensor, original_size = _preprocess_image(image_input)
    original_width, original_height = original_size
    target_width, target_height = target_size

    scale_width = original_width / target_width
    scale_height = original_height / target_height
    scale = (scale_width + scale_height) / 2.0

    with torch.no_grad():
        lr_image_tensor, _ = model(input_tensor, scale=scale)
    
    downscaled_image_np = _postprocess_image(lr_image_tensor)
    return downscaled_image_np

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


# 新增圖像填充函數
def pad_to_square_and_downscale_friendly(image_np: np.ndarray, target_base_dim: int = 224, fill_color=(0, 0, 0)) -> np.ndarray:
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
    max_dim = max(height, width)

    # 計算新的正方形邊長，使其是 target_base_dim 的倍數且大於等於 max_dim
    # 例如：如果 max_dim=500, target_base_dim=224
    # math.ceil(500 / 224) = math.ceil(2.23) = 3
    # padded_side = 3 * 224 = 672
    target_mult = math.ceil(max_dim / target_base_dim)
    padded_side = int(target_mult * target_base_dim)

    # 計算填充後的圖像應該放在新圖像的哪個位置 (置中)
    pad_h = (padded_side - height) // 2
    pad_w = (padded_side - width) // 2

    # 創建一個新的空白圖像，用指定顏色填充
    padded_image = np.full((padded_side, padded_side, 3), fill_color, dtype=np.uint8)

    # 將原始圖像複製到新圖像的中心
    padded_image[pad_h:pad_h + height, pad_w:pad_w + width] = image_np

    # print(f"Image padded from {width}x{height} to {padded_side}x{padded_side} for downscaling to {target_base_dim}x{target_base_dim}.")

    return padded_image

# 將圖像下採樣到 224x224 的 Lanczos 包裝函數
def downscale_lanczos_224(image_input_np: np.ndarray) -> np.ndarray:
    """
    使用 Lanczos 插值將圖像填充為正方形並下採樣到 224x224。
    Args:
        image_input_np (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。
    Returns:
        numpy.ndarray: 處理後的圖像 (OpenCV BGR 格式)。
    """
    padded_img_np = pad_to_square_and_downscale_friendly(image_input_np, target_base_dim=224)
    return _downscale_lanczos_raw(padded_img_np, target_size=(224, 224))

# MATLAB 引擎管理器，使用單例模式和上下文管理器
class MatlabEngineManager:
    _instance = None
    _engine = None
    _project_root = None # 儲存專案根目錄，只初始化一次

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MatlabEngineManager, cls).__new__(cls)
            # 在此處確定專案根目錄，只需一次
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(current_dir, os.pardir)
            cls._project_root = os.path.join(src_dir, os.pardir)
        return cls._instance

    def __enter__(self):
        if self._engine is None:
            print("Starting MATLAB engine...")
            try:
                self._engine = matlab.engine.start_matlab()
                l0_downscaling_path = os.path.join(self._project_root, 'L-0 downscaling')
                self._engine.addpath(l0_downscaling_path)
                print(f"MATLAB engine started and L-0 downscaling path added: {l0_downscaling_path}")
            except Exception as e:
                print(f"Failed to start MATLAB engine or add path: {e}")
                self._engine = None # 確保引擎狀態為 None
                raise # 重新拋出異常
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._engine is not None:
            print("Quitting MATLAB engine...")
            try:
                self._engine.quit()
            except Exception as e:
                print(f"Error quitting MATLAB engine: {e}")
            finally:
                self._engine = None
    
    def get_engine(self):
        if self._engine is None:
            raise RuntimeError("MATLAB engine is not running. Please use 'with MatlabEngineManager():' to manage the engine.")
        return self._engine


# 新增 L0 梯度最小化下採樣的工廠函數
def get_l0_downscaler_224(lambda_val: float = 2e-4, kappa_val: float = 2):
    """
    返回一個 L0 梯度最小化下採樣函數的包裝器，該包裝器與 preprocess_folder 兼容。
    Args:
        lambda_val (float): L0 梯度最小化中的 lambda 參數。
        kappa_val (float): L0 梯度最小化中的 kappa 參數。
    Returns:
        Callable[[numpy.ndarray], numpy.ndarray]: L0 下採樣函數。
    """
    def l0_downscale_wrapper(image_input_np: np.ndarray) -> np.ndarray:
        # 獲取 MATLAB 引擎實例
        manager = MatlabEngineManager()
        eng = manager.get_engine()

        current_img_np = image_input_np.copy()
        current_height, current_width = current_img_np.shape[:2]
        max_dim = max(current_height, current_width)
        target_final_dim = 224

        # Case 2: 長邊邊長超過 224 * 16 的情況
        while max_dim > target_final_dim * 16:
            print(f"L0 Pre-Downscaling (Coarse): Current max_dim {max_dim} > {target_final_dim * 16}. Applying L0 factor 4.")
            downscale_factor_step = 4
            
            current_img_np = _downscale_l0_raw(current_img_np,
                                                downscaling_factor=downscale_factor_step,
                                                eng=eng,
                                                lambda_val=lambda_val, kappa_val=kappa_val)
            current_height, current_width = current_img_np.shape[:2]
            max_dim = max(current_height, current_width)
            print(f"L0 Pre-Downscaling Result: {current_width}x{current_height}")

        # Case 1: 長邊邊長不超過 224 * 16，或已縮小到此範圍。
        if max_dim > target_final_dim:
            print(f"L0 Main Downscaling (Fine): Current max_dim {max_dim} > {target_final_dim}. Finding optimal L0 factor.")
            
            optimal_downscale_factor = 2
            for f in range(2, 17):
                if max_dim / f <= target_final_dim:
                    optimal_downscale_factor = f
                    break
            
            print(f"Optimal L0 factor found: {optimal_downscale_factor}.")
            
            current_img_np = _downscale_l0_raw(current_img_np,
                                                downscaling_factor=optimal_downscale_factor,
                                                eng=eng,
                                                lambda_val=lambda_val, kappa_val=kappa_val)
            current_height, current_width = current_img_np.shape[:2]
            max_dim = max(current_height, current_width)
            print(f"L0 Main Downscaling Result: {current_width}x{current_height}. (Should be <= {target_final_dim})")
        
        # 最終步驟：填充到 224x224
        final_processed_img = pad_to_square_and_downscale_friendly(current_img_np, target_base_dim=target_final_dim)
        
        return final_processed_img
    return l0_downscale_wrapper

# 新增 L0 下採樣方法
def _downscale_l0_raw(image_input: np.ndarray, downscaling_factor: int, eng, lambda_val: float = 2e-4, kappa_val: float = 2):
    """
    使用 L0 梯度最小化下採樣算法將圖像下採樣。此函數執行單次下採樣。

    Args:
        image_input (numpy.ndarray): 輸入圖像 (OpenCV BGR 格式的 NumPy 陣列)。
        downscaling_factor (int): L0 下採樣的因子 (介於 2 到 16 之間)。
        eng: 已啟動的 MATLAB 引擎實例。
        lambda_val (float): L0 梯度最小化中的 lambda 參數。
        kappa_val (float): L0 梯度最小化中的 kappa 參數。

    Returns:
        numpy.ndarray: 下採樣後的圖像 (OpenCV BGR 格式)。
    """
    temp_input_path = None
    temp_output_path = None
    processed_img_np = None

    try:
        if not isinstance(image_input, np.ndarray):
            raise TypeError("Input image_input must be a numpy.ndarray.")

        current_img_np = image_input.copy() # 使用副本，避免修改原始輸入
        current_height, current_width = current_img_np.shape[:2]

        downscaling_factor_step = downscaling_factor # 直接使用傳入的因子
        
        if downscaling_factor_step < 2:
            print(f"L0 Downscaling: Skipping L0 step as factor {downscaling_factor_step} is less than 2.")
            return image_input
        
        # 確保下採樣因子在 MATLAB 函數的推薦範圍 (2-16) 內
        downscaling_factor_step = int(min(downscaling_factor_step, 16.0))
        downscaling_factor_step = max(downscaling_factor_step, 2)

        # print(f"L0 Downscaling Step: Current size {current_width}x{current_height}. Applying factor {downscaling_factor_step}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input_file:
            temp_input_path = temp_input_file.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output_file:
            temp_output_path = temp_output_file.name

        cv2.imwrite(temp_input_path, current_img_np)

        # 呼叫 MATLAB 函數進行一步下採樣
        eng.l0_downscale_auto(temp_input_path, float(downscaling_factor_step), temp_output_path, float(lambda_val), float(kappa_val), nargout=0)

        # 讀取 MATLAB 處理後的圖像
        processed_img_np = cv2.imread(temp_output_path)
        if processed_img_np is None:
            raise IOError(f"Could not read processed image from {temp_output_path} after L0 step with factor {downscaling_factor_step}. Input was {temp_input_path}")
        
    except Exception as e:
        print(f"Error during L0 downscaling: {e}")
        processed_img_np = image_input # 在出錯時返回原始圖像
    finally:
        # 清理任何剩餘的臨時文件
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    
    return processed_img_np
