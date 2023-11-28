import pdb

import numpy as np
import PIL
from avalanche.benchmarks import AvalancheDataset, SplitTinyImageNet
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.benchmarks.utils.ffcv_support import (
    HybridFfcvLoader,
    enable_ffcv,
)
from torch.testing import assert_close
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from continualUtils.benchmarks import SplitClickMe


def test_load_splitimagenet(device, tmpdir):
    """Test loading SplitImageNet with FFCV"""
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


def test_load_splitclickme():
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


def test_ffcv_clickme(device, tmpdir):
    """Test loading the SplitClickMe benchmark"""
    dataset_root = "/mnt/datasets/clickme"
    batch_size = 16
    num_workers = 2
    # benchmark = SplitTinyImageNet(
    #     n_experiences=10, dataset_root="/mnt/datasets/tinyimagenet", seed=42
    # )
    benchmark = SplitClickMe(
        n_experiences=2,
        root=dataset_root,
        seed=79,
        dummy=True,
        return_task_id=True,
        shuffle=True,
        class_ids_from_zero_in_each_exp=True,
        fixed_class_order=list(range(0, 1000)),
    )

    enable_ffcv(
        benchmark=benchmark,
        write_dir=f"{tmpdir}/ffcv_clickme",
        device=device,
        ffcv_parameters=dict(
            num_workers=num_workers,
            write_mode="proportion",
            compress_probability=0.25,
            jpeg_quality=90,
        ),
        force_overwrite=False,
        print_summary=True,  # Better keep this true on non-benchmarking code
    )

    all_train_dataset = [x.dataset for x in benchmark.train_stream]
    avl_set = AvalancheDataset(all_train_dataset)
    avl_set = avl_set.train()

    ffcv_loader = HybridFfcvLoader(
        dataset=avl_set,
        batch_sampler=BatchSampler(
            SequentialSampler(avl_set),
            batch_size=batch_size,
            drop_last=True,
        ),
        ffcv_loader_parameters=dict(
            num_workers=num_workers,
        ),
        device=device,
        print_ffcv_summary=True,
    )
