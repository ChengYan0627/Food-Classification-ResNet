from matplotlib import pyplot as plt

def single_image(image_path: str, title=None):
    """
    Plots an image from the given path
    """
    image = plt.imread(image_path)
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()