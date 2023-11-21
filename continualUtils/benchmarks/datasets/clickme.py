"""
Dataset access code adapted from the Harmonization project:
https://github.com/serre-lab/Harmonization/blob/main/harmonization/common/clickme_dataset.py
"""
import json
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
from avalanche.benchmarks.datasets import ImageNet
from avalanche.benchmarks.utils import make_classification_dataset
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import (
    gaussian_blur,
    pil_to_tensor,
    resize,
)

from continualUtils.benchmarks.datasets.preprocess import preprocess_input

# Dataset hosting info
CLICKME_BASE_URL = (
    "https://connectomics.clps.brown.edu/tf_records/clicktionary_files/"
)
TRAIN_ZIP = "clickme_train.zip"
TEST_ZIP = "clickme_test.zip"
VAL_ZIP = "clickme_val.zip"

LOCAL_PATH = "~/datasets/clickme/"

HEATMAP_INDEX = 2
TOKEN_INDEX = 3


def make_clickme_style_imagenet_dataset(
    root: str, split: Literal["train", "val"]
):
    """Returns ClickMeImageNetWrapperDataset as an Avalanche Dataset"""
    dataset = ClickMeImageNetWrapperDataset(root=root, split=split)
    return make_classification_dataset(dataset)


def make_clickme_dataset(root: str, split: Literal["train", "val", "test"]):
    """Returns ClickMe as an Avalanche Dataset"""
    dataset = ClickMeDataset(root=root, split=split)
    return make_classification_dataset(dataset)


class ClickMeImageNetWrapperDataset(datasets.ImageNet):
    """Dataset generator that wraps around ImageNet to return ClickMe style
    dataset
    """

    map_placeholder: Tensor = torch.empty((1, 224, 224), dtype=torch.float)

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            **kwargs
        )

    def __getitem__(self, index: int):
        # Retrieve the image and label from the ImageNet dataset
        image, label = super().__getitem__(index)

        image = preprocess_input(
            torch.permute(pil_to_tensor(image), (1, 2, 0))
        ).float()
        image = torch.permute(image, (2, 0, 1))
        image = resize(image, size=(224, 224), antialias=False)  # type: ignore

        # Extend the dataset to return ClickMe style data
        heatmap = ClickMeImageNetWrapperDataset.map_placeholder
        token = torch.tensor(0).float()

        return image, label, heatmap, token


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
        target_path = self.full_path.joinpath("metadata.json")
        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                self.targets = [metadata[file.name] for file in self.files]
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

    def __getitem__(self, index):
        """
        Returns a batch of images, labels, and heatmaps as Torch tensors
        Note: due to how Avalanche processes the batch, we must preserve the images, labels, and heatmaps order.
        """
        data = np.load(str(self.files[index]))
        np_img = data["image"]
        np_heatmap = data["heatmap"]
        np_label = data["label"]

        np_img = preprocess_input(np_img)
        image = torch.from_numpy(np.transpose(np_img, (2, 0, 1))).float()
        image = resize(image, size=224, antialias=False)  # type: ignore

        # Process heatmap
        heatmap = torch.from_numpy(np_heatmap).float()
        heatmap = resize(heatmap.unsqueeze(0), size=64, antialias=False)  # type: ignore
        heatmap = gaussian_blur(heatmap, kernel_size=(11, 11), sigma=(11, 11))  # type: ignore
        # ValueError: kernel_size should have odd and positive integers. Got (10, 10)

        heatmap = resize(heatmap, size=224, antialias=False)  # type: ignore

        label = np_label
        token = self.tokens[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, heatmap, token
