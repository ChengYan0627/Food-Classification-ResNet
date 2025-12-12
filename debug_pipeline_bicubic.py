import sys
import os
import cv2
import numpy as np
import shutil

# Add src to path so we can import models and project_utils
sys.path.insert(0, './src')

try:
    from models.classifier import FoodClassifier
    from project_utils import calculate_metrics, preprocess_folder
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

from PIL import Image

# ==========================================
# Local Helper Functions (Replicating logic)
# ==========================================

def _center_crop(image_np: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """
    Center crop to crop_size x crop_size, no padding.
    """
    h, w = image_np.shape[:2]
    if h < crop_size or w < crop_size:
        return image_np
    
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return image_np[y0:y0 + crop_size, x0:x0 + crop_size]

def Bicubic(image_input_np: np.ndarray) -> np.ndarray:
    """
    Debug: Bicubic downscaling to verify pipeline.
    Replaces complex transformer/DPID logic with simple OpenCV bicubic resize.
    
    Steps:
    1. Resize short side to 256 using Bicubic interpolation.
    2. Center crop to 224x224.
    """
    h, w = image_input_np.shape[:2]
    if h == 0 or w == 0:
        return image_input_np
        
    # Target short side = 256 (same as Lanczos/DPID pipeline target)
    target_short = 256
    
    if h <= w:
        new_h = target_short
        new_w = int(round(w * target_short / float(h)))
    else:
        new_w = target_short
        new_h = int(round(h * target_short / float(w)))
        
    # Resize using Bicubic (INTER_CUBIC)
    resized = cv2.resize(image_input_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Center crop to 224x224
    cropped = _center_crop(resized, crop_size=224)
    
    return cropped

def PIL_Bicubic(image_input_np: np.ndarray) -> np.ndarray:
    """
    Debug: PIL-based Bicubic downscaling.
    Simulates Hugging Face AutoImageProcessor behavior more closely.
    
    Steps:
    1. Convert NumPy (BGR) -> PIL (RGB).
    2. Resize short side to 256 using PIL.Image.BICUBIC.
    3. Center crop to 224x224.
    4. Convert back to NumPy (BGR) for compatibility.
    """
    # Convert OpenCV BGR to PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(image_input_np, cv2.COLOR_BGR2RGB))
    
    w, h = image_pil.size
    target_short = 256
    
    if h <= w:
        new_h = target_short
        new_w = int(round(w * target_short / float(h)))
    else:
        new_w = target_short
        new_h = int(round(h * target_short / float(w)))
        
    # Resize using PIL's Bicubic
    resized_pil = image_pil.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # Center crop
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    
    cropped_pil = resized_pil.crop((left, top, right, bottom))
    
    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)

def PIL_DirectResize(image_input_np: np.ndarray) -> np.ndarray:
    """
    Hypothesis 1: Direct Resize to 224x224 (Squish), no cropping.
    Some models prefer seeing the whole image even if distorted.
    """
    image_pil = Image.fromarray(cv2.cvtColor(image_input_np, cv2.COLOR_BGR2RGB))
    resized_pil = image_pil.resize((224, 224), resample=Image.BICUBIC)
    return cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

def PIL_ResizeShort224(image_input_np: np.ndarray) -> np.ndarray:
    """
    Hypothesis 2: Resize short side to EXACTLY 224 (not 256), then Center Crop.
    This minimizes the 'zoom in' effect.
    """
    image_pil = Image.fromarray(cv2.cvtColor(image_input_np, cv2.COLOR_BGR2RGB))
    w, h = image_pil.size
    target_short = 224  # Change from 256 to 224
    
    if h <= w:
        new_h = target_short
        new_w = int(round(w * target_short / float(h)))
    else:
        new_w = target_short
        new_h = int(round(h * target_short / float(w)))
        
    resized_pil = image_pil.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # Center crop 224 (which is trivial on the short side now)
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    
    cropped_pil = resized_pil.crop((left, top, right, bottom))
    return cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)

def PIL_Lanczos(image_input_np: np.ndarray) -> np.ndarray:
    """
    Hypothesis 3: Use LANCZOS filter instead of BICUBIC.
    Lanczos is higher quality and sharper.
    """
    image_pil = Image.fromarray(cv2.cvtColor(image_input_np, cv2.COLOR_BGR2RGB))
    w, h = image_pil.size
    target_short = 256
    
    if h <= w:
        new_h = target_short
        new_w = int(round(w * target_short / float(h)))
    else:
        new_w = target_short
        new_h = int(round(h * target_short / float(w)))
        
    # Use LANCZOS
    resized_pil = image_pil.resize((new_w, new_h), resample=Image.LANCZOS)
    
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    
    cropped_pil = resized_pil.crop((left, top, right, bottom))
    return cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)

def PIL_DirectResize_Lanczos(image_input_np: np.ndarray) -> np.ndarray:
    """
    Hypothesis 4: Direct Resize to 224x224 (Squish) using LANCZOS filter.
    Combines the winning 'Squish' strategy with higher quality interpolation.
    """
    image_pil = Image.fromarray(cv2.cvtColor(image_input_np, cv2.COLOR_BGR2RGB))
    resized_pil = image_pil.resize((224, 224), resample=Image.LANCZOS)
    return cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# In-Memory Evaluation Logic
# ==========================================

from torch.utils.data import Dataset, DataLoader
import torch

class InMemoryDataset(Dataset):
    """
    Dataset that performs PIL Bicubic resizing on-the-fly without saving to disk.
    This simulates the "purest" pipeline closest to Hugging Face's internal behavior.
    """
    def __init__(self, data_dir: str, processor, id_to_label: dict):
        self.data_dir = data_dir
        self.processor = processor
        
        # Reverse mapping from label name to ID
        self.label_map = {int(k): v for k, v in id_to_label.items()}
        self.label_to_id = {v: k for k, v in self.label_map.items()}
        
        self.images = []
        self.labels = []
        self.names = []

        # Load file list
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.JPG', '.jpeg')):
                name = filename.lower()
                stem = os.path.splitext(name)[0]
                label = '_'.join(stem.split('_')[:-1])
                if label in self.label_to_id:
                    self.images.append(os.path.join(data_dir, filename))
                    self.labels.append(self.label_to_id[label])
                    self.names.append(name)
        
        print(f"InMemoryDataset loaded {len(self.images)} images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_id = int(self.labels[idx])
        
        # 1. Load Original High-Res Image (PIL)
        image_pil = Image.open(image_path).convert("RGB")
        
        # 2. Perform PIL Bicubic Resize + Center Crop (In-Memory)
        w, h = image_pil.size
        target_short = 256
        
        if h <= w:
            new_h = target_short
            new_w = int(round(w * target_short / float(h)))
        else:
            new_w = target_short
            new_h = int(round(h * target_short / float(w)))
            
        resized_pil = image_pil.resize((new_w, new_h), resample=Image.BICUBIC)
        
        left = (new_w - 224) / 2
        top = (new_h - 224) / 2
        right = (new_w + 224) / 2
        bottom = (new_h + 224) / 2
        cropped_pil = resized_pil.crop((left, top, right, bottom))
        
        # 3. Pass directly to Processor (Only for Normalize/Rescale, NO resizing)
        # We assume processor.do_resize = False was set globally
        processed = self.processor(images=cropped_pil, return_tensors="pt")
        pixel_values = processed.pixel_values.squeeze(0)
        
        return pixel_values, label_id, self.names[idx]

def run_in_memory_test(model, dataset_path):
    """
    Runs evaluation using the InMemoryDataset.
    """
    print(f"\nPreprocessing images In-Memory (PIL Bicubic) -> Model...")
    
    # Create dataset & dataloader
    dataset = InMemoryDataset(
        data_dir=dataset_path, 
        processor=model.processor, 
        id_to_label=model.labels
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    model.model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            outputs = model.model(pixel_values=images)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    metrics = calculate_metrics(true_labels=all_labels, pred_labels=all_preds)
    print(f"\nSUCCESS! In-Memory execution finished.")
    print(f"Metrics for In-Memory PIL: {metrics}")
    return metrics

# ==========================================
# Main Test Execution
# ==========================================

def main():
    print("Initializing FoodClassifier Model...")
    try:
        model = FoodClassifier()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Critical Step: Disable model's internal resizing/cropping
    # because our 'Bicubic' function already produces the final 224x224 input.
    print("Configuring model processor (disable resize/crop)...")
    model.processor.do_resize = False
    model.processor.do_center_crop = False
    
    dataset_path = './data/raw'
    output_base_path = './data/debug_preprocessed'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return

    # --- New Step: Run In-Memory Test ---
    run_in_memory_test(model, dataset_path)
    
    print(f"\n--- Starting Debug Pipeline with Multiple PIL Strategies ---")
    
    # List of strategies to test
    strategies = [
        PIL_DirectResize,          # Winning Baseline (Bicubic) (~57.5%)
        PIL_DirectResize_Lanczos,  # New Challenger (Lanczos)
    ]

    for func in strategies:
        out_dir = output_base_path + func.__name__
        
        # Clean up previous run if exists
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            
        print(f"\nPreprocessing images from {dataset_path} to {out_dir}...")
        print(f"Using algorithm: {func.__name__}")
        
        try:
            preprocess_folder(func, input_dir=dataset_path, output_dir=out_dir)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            continue
        
        print(f"Running prediction on {out_dir}...")
        
        try:
            results = model.predict_folder(out_dir)
            metrics = calculate_metrics(**results)
            print(f"Metrics for {func.__name__}: {metrics}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

    print("-" * 50)
    print("\nTest finished.")

if __name__ == "__main__":
    main()

