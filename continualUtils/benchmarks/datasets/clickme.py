"""
Dataset access code adapted from the Harmonization project:
https://github.com/serre-lab/Harmonization/blob/main/harmonization/common/clickme_dataset.py
"""
import json
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_clickme_style_imagenet_dataset(
    root: str, split: Literal["train", "val"]
):
    """Returns ClickMeImageNetWrapperDataset as an Avalanche Dataset"""
    dataset = ClickMeImageNetWrapperDataset(root=root, split=split)
    return _make_taskaware_classification_dataset(dataset)


def make_clickme_dataset(
    root: str, split: Literal["train", "val", "test", "dtrain", "dtest"]
):
    """Returns ClickMe as an Avalanche Dataset"""
    dataset = ClickMeDataset(root=root, split=split)
    return _make_taskaware_classification_dataset(dataset)


class ClickMeImageNetWrapperDataset(datasets.ImageNet):
    """Dataset generator that wraps around ImageNet to return ClickMe style dataset."""

    map_placeholder: Tensor = torch.empty((1, 224, 224), dtype=torch.float)

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        # If no transform is provided, define the default
        if transform is None:
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize((224, 224), antialias=True),
                    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

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

        # Heatmap and token are placeholders
        heatmap = ClickMeImageNetWrapperDataset.map_placeholder
        token = torch.tensor([0])

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
        target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.split = split

        self.split_dir = "clickme_" + split + "/"

        self.full_path = Path(self.root).joinpath(self.split_dir)
        self.files = list(self.full_path.glob("*.npz"))

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

        # Define composed transforms for images
        self.transform = (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize((224, 224), antialias=True),
                    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
            if transform is None
            else transform
        )

        # Define composed transforms for heatmap
        self.heatmap_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((64, 64), antialias=True),
                v2.GaussianBlur(kernel_size=(7, 7), sigma=(7, 7)),
                v2.Resize((224, 224), antialias=True),
            ]
        )

        # Define composed transforms for labels if needed
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.targets)

    def __getitem__(self, index):
        data = np.load(str(self.files[index]))
        image = data["image"]
        label = data["label"]
        heatmap = data["heatmap"]

        # Transform images
        if self.transform is not None:
            image = self.transform(image)

        # Apply any transformations to labels
        if self.target_transform is not None:
            label = self.target_transform(label)

        # Process heatmap
        heatmap = heatmap[..., np.newaxis]
        if self.heatmap_transform is not None:
            heatmap = self.heatmap_transform(heatmap)

        token = torch.tensor([1])

        return image, label, heatmap, token
