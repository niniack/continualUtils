"""
Dataset access code adapted from the Harmonization project:
https://github.com/serre-lab/Harmonization/blob/main/harmonization/common/clickme_dataset.py
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from typing import Optional, Callable, List

from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur, resize

from avalanche.benchmarks.utils import make_classification_dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Dataset hosting info
CLICKME_BASE_URL = "https://connectomics.clps.brown.edu/tf_records/clicktionary_files/"
TRAIN_ZIP = "clickme_train.zip"
TEST_ZIP = "clickme_test.zip"
VAL_ZIP = "clickme_val.zip"

LOCAL_PATH = "~/datasets/clickme/"

HEATMAP_INDEX = 2
TOKEN_INDEX = 3


def initialize_dataset(root: str, split: str):
    """Initializes an Avalanche Dataset

    Args:
        root (str): path for directory
        split (str): train, test, or val split

    Returns:
        AvalancheDataset: an Avalanche Classification Dataset instance
    """
    base_ds = ClickMeDataset(root=root, split=split)

    return make_classification_dataset(base_ds)


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


class ClickMeDataset(Dataset):
    """Dataset generator for the ClickMe dataset. Returns torch tensors.

    The ClickMe dataset contains 196,499 unique images.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.split = split

        self.split_dir = "clickme_" + split + "/"
        self.full_path = Path(self.root).joinpath(self.split_dir)
        self.files = list(self.full_path.glob("*.npz"))

        self.transform = transform
        # Load targets
        target_path = self.full_path.joinpath("targets.pkl")
        if target_path.exists():
            with open(target_path, "rb") as file:
                self.targets = pickle.load(file)
        else:
            self.targets = [
                int(np.load(str(self.files[x]))["label"])
                for x in range(len(self.files))
            ]
        self.tokens = torch.ones(len(self.files))

    def __len__(self):
        """
        Returns the size of the dataset
        """

        return len(self.targets)

    def __getitem__(self, idx):
        """
        Returns a batch of images, labels, and heatmaps as Torch tensors

        Note: due to how Avalanche processes the batch, we must preserve the images, labels, and heatmaps order.
        """

        np_img = np.load(str(self.files[idx]))["image"]
        np_heatmap = np.load(str(self.files[idx]))["heatmap"]
        np_label = np.load(str(self.files[idx]))["label"]

        np_img = preprocess_input(np_img)
        image = torch.from_numpy(np.transpose(np_img, (2, 0, 1))).float()
        image = resize(image, size=224)

        # Process heatmap
        heatmap = torch.from_numpy(np_heatmap).float()
        heatmap = resize(heatmap.unsqueeze(0), size=64)
        heatmap = gaussian_blur(heatmap, kernel_size=(9, 9), sigma=(9, 9))
        heatmap = resize(heatmap, size=224)

        label = np_label
        token = self.tokens[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, heatmap, token
