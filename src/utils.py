import warnings
import sklearn.exceptions
import os
from PIL import Image
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def calculate_metrics(true_labels: dict, pred_labels: dict, **kwargs):
    """
    Calculate accuracy and F1-score.
    """
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    return {"accuracy": accuracy, "f1_score": f1}   

def preprocess_folder(function: callable, input_dir: str, output_dir: str):
    """
    Preprocess all the images in the input folder and save the results
    """

    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        img = Image.open(input_path)
        processed_img = function(img)
        processed_img.save(output_path)

        del img
        del processed_img   

    return