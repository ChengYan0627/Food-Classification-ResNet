import matplotlib.pyplot as plt
import warnings
import sklearn.exceptions
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def plot_image(image_path: str, title=None):
    """
    Plots an image from the given path
    """
    image = plt.imread(image_path)
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.show()

def calculate_metrics(true_labels: dict, pred_labels: dict, **kwargs):
    """
    Calculate accuracy and F1-score.
    """
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    return {"accuracy": accuracy, "f1_score": f1}   