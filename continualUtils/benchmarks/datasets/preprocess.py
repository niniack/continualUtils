import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

"""
Preprocess function from:
https://github.com/serre-lab/Harmonization/blob/main/harmonization/models/preprocess.py
"""


def preprocess_input(images):
    """Preprocesses images for the harmonized models.
    The images are expected to be in RGB format with values in the range [0, 255].

    Args:
        images (Tensor or numpy array): image to be processed

    Returns:
        Tensor or numpy array: Preprocessed images
    """
    images = images / 255.0

    images = images - IMAGENET_MEAN
    images = images / IMAGENET_STD

    return images
