import numpy as np
from avalanche.benchmarks import SplitTinyImageNet
from torch.testing import assert_close

from continualUtils.benchmarks import SplitClickMe
from continualUtils.benchmarks.datasets.preprocess import preprocess_input


def test_load_tiny_imagenet(logger):
    split_tiny_imagenet = SplitTinyImageNet(
        n_experiences=10, dataset_root="/mnt/datasets/tinyimagenet", seed=42
    )
    logger.debug(dir(split_tiny_imagenet))


def test_normalize_np_image():
    """Test normalizing a numpy image"""

    # Create a 3D tensor
    np_img = np.array(
        [
            [[255.0, 255.0, 255.0], [255.0, 255.0, 255.0]],
            [[255.0, 255.0, 255.0], [255.0, 255.0, 255.0]],
        ]
    )

    # Call the normalization function
    normalized_img = preprocess_input(np_img)

    # Define expectation
    expected_img = np.array(
        [
            [[2.249, 2.429, 2.640], [2.249, 2.429, 2.640]],
            [[2.249, 2.429, 2.640], [2.249, 2.429, 2.640]],
        ]
    )

    # Assert that the output is as expected
    assert_close(normalized_img, expected_img, rtol=1e-03, atol=1e-03)


def test_load_splitclickme(logger):
    """Test loading the SplitClickMe benchmark"""
    split_clickme = SplitClickMe(
        n_experiences=10,
        root="/mnt/datasets/clickme",
        seed=42,
    )

    assert split_clickme.train_stream is not None
    assert split_clickme.test_stream is not None
    assert split_clickme.val_stream is not None

    val_exp = next(iter(split_clickme.val_stream), None)
    assert val_exp is not None

    assert hasattr(val_exp, "classes_in_this_experience")

    ds = split_clickme.test_stream[0].dataset
    image, label, heatmap, token, task = ds[0]
