from pathlib import Path
from typing import Sequence, Optional, Union, Any

from torchvision import transforms

from avalanche.benchmarks import nc_benchmark, NCScenario
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from datasets.clickme import ClickMeDataset, initialize_dataset


def SplitClickMe(
    n_experiences: int,
    root: str,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = False,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Union[str, Path] = None,
    dummy: bool = False,
) -> NCScenario:
    # DEBUG for faster loading of the dataset
    # TODO fix naming, don't use dummy dirs
    if dummy:
        clickme_train = ClickMeDataset(root=root, split="val")
        clickme_test = ClickMeDataset(root=root, split="test")
    # TODO fix naming, don't use val, use train!
    else:
        clickme_train = initialize_dataset(root=root, split="train")
        clickme_test = initialize_dataset(root=root, split="test")

    return nc_benchmark(
        train_dataset=clickme_train,
        test_dataset=clickme_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        per_exp_classes=None,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
