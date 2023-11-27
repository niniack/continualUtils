import numpy as np
from avalanche.benchmarks import SplitTinyImageNet
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.benchmarks.utils.ffcv_support import enable_ffcv
from torch.testing import assert_close

from continualUtils.benchmarks import SplitClickMe
from continualUtils.benchmarks.datasets.preprocess import preprocess_input


def test_load_splitimagenet(device, tmpdir):
    split_tiny_imagenet = SplitTinyImageNet(
        n_experiences=10, dataset_root="/mnt/datasets/tinyimagenet", seed=42
    )
    benchmark_type = "tinyimagenet"
    enable_ffcv(
        benchmark=split_tiny_imagenet,
        write_dir=f"{tmpdir}/ffcv_test_{benchmark_type}",
        device=device,
        ffcv_parameters=dict(num_workers=8),
        print_summary=True,
    )


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


def test_load_splitclickme(logger, device, tmpdir):
    """Test loading the SplitClickMe benchmark"""

    ds_root = "/mnt/datasets/clickme"

    split_clickme = SplitClickMe(
        n_experiences=10, root=ds_root, seed=42, dummy=False
    )

    assert split_clickme.train_stream is not None
    assert split_clickme.test_stream is not None
    assert split_clickme.val_stream is not None

    val_exp = next(iter(split_clickme.val_stream), None)
    assert val_exp is not None

    assert hasattr(val_exp, "classes_in_this_experience")

    dataloader = TaskBalancedDataLoader(
        split_clickme.train_stream[0].dataset,
        oversample_small_groups=True,
    )

    batch = next(iter(dataloader))

    assert len(batch) is 5

    image, label, heatmap, token, task = batch
