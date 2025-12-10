# ECE253 — Food Image Classification
## By Vincent Kao, Cheng-Yan Juang, Tien-Hao Chen

This project implements an image classification pipeline focused on food images (based on the Food-101 dataset and a [pre-trained Siglip model](https://huggingface.co/prithivMLmods/Food-101-93M)). It contains preprocessing algorithms (low-light correction, deblurring, downscaling), a model wrapper for inference and evaluation, and a simple trainer for fine-tuning.

## Project structure

```
📦 
├─ data
│  ├─ raw                    # Raw collected images
│  └─ preprocessed           # Temporary preprocessed images
│
├─ src
│  ├─ models
│  │  ├─ classifier.py       # `FoodClassifier` model wrapper and `FoodImageDataset` Dataset class
│  │  └─ trainer.py          # Model fine-tuning stuff
│  │ 
│  ├─ preprocessing
│  │  ├─ deblurr.py          # Deblurr algorithms
│  │  └─ lowlight.py         # Lowlight-enhancement algorithms
│  │  └─ downscaling.py      # Downscaling algorithms
│  │
│  ├─ plot.py                # Plotting helper function
│  └─ utils.py               # Useful functions for calculating metrics and preprocessing
│
├─ project.ipynb             # Notebook with experiments
│  proposal.pdf              # Project proposal
└─ README.md                 # This file
```


## How to run

In Jupyter Notebook:

1. Open `project.ipynb` in VS Code or Jupyter.
2. Run the cells in order:
   - Initializing `FoodClassifier()` and predicting a single image.
   - Running `predict_folder` dataset from `dataset\raw`.
   - Applying preprocessing functions and evaluating results.
   - Fine-tuning via `models.trainer.fine_tune`.

Notes:
- Dataset images need to include a label matching the Food-101 style, with a number id. For example, `hamburger_01.png`.

## External packages used

The project uses the following external Python packages:

- `torch` (PyTorch) — model and DataLoader, training and inference.
- `transformers` — `AutoImageProcessor` and `SiglipForImageClassification`.
- `Pillow` (`PIL`) — image I/O.
- `opencv-python` (`cv2`) — preprocessing helpers.
- `numpy` — numerical operations in preprocessing.
- `scipy` — signal processing (used in deblurring functions).
- `scikit-learn` — evaluation metrics (F1 and other helpers).
- `matplotlib` — plotting utilities used by `src/plot.py`.
