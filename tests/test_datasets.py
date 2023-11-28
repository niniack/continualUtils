import pdb

import numpy as np
import PIL
from avalanche.benchmarks.utils.dataset_traversal_utils import (
    single_flat_dataset,
)

from continualUtils.benchmarks.datasets.clickme import (
    ClickMeDataset,
    ClickMeImageNetWrapperDataset,
    make_combined_clickme_dataset,
)


def test_combined_dataset():
    """Test loading the CombinedClickMeDataset"""

    dataset = make_combined_clickme_dataset(
        imagenet_root="/mnt/datasets/fake_imagenet",
        clickme_root="/mnt/datasets/clickme",
        imagenet_split="train",
        clickme_split="dtrain",
    )

    flat_set = single_flat_dataset(dataset)
    assert flat_set is not None


def test_wrapper_dataset():
    """Test loading ClickMeImageNetWrapperDataset with no transforms enabled"""
    dataset = ClickMeImageNetWrapperDataset(
        root="/mnt/datasets/fake_imagenet",
        split="train",
        apply_transform=False,
    )

    flat_set = single_flat_dataset(dataset)
    assert flat_set is not None

    # Call __getitem__
    assert len(dataset[0]) == 4
    image, label, heatmap, token = dataset[0]

    assert isinstance(image, PIL.Image.Image)  # type: ignore
    assert isinstance(label, int)
    assert isinstance(heatmap, np.ndarray)
    assert isinstance(token, int)

    assert heatmap.shape == (256, 256, 1)


def test_clickme_dataset():
    """Test loading ClickMeDataset with no transforms enabled"""
    dataset = ClickMeDataset(
        root="/mnt/datasets/clickme",
        split="dtrain",
        apply_transform=False,
    )

    flat_set = single_flat_dataset(dataset)
    assert flat_set is not None

    # Call __getitem__
    assert len(dataset[0]) == 4
    image, label, heatmap, token = dataset[0]

    assert isinstance(image, PIL.Image.Image)  # type: ignore
    assert isinstance(label, int)
    assert isinstance(heatmap, np.ndarray)
    assert isinstance(token, int)

    assert heatmap.shape == (256, 256, 1)
