from typing import Any, Optional, Sequence

from avalanche.benchmarks import NCScenario, nc_benchmark

from continualUtils.benchmarks.datasets.clickme import ClickMeDataset


def SplitClickMe(  # pylint: disable=C0103
    n_experiences: int,
    root: str,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = False,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dummy: bool = False,
) -> NCScenario:
    """Returns a split version of the ClickMe dataset

    :param n_experiences: The number of incremental experience. This is not used
        when using multiple train/test datasets with the ``one_dataset_per_exp``
        parameter set to True.
    :param root: Root of the dataset, where train and test are accessible
    :param return_task_id: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences., defaults to False
    :param seed: If ``shuffle`` is True and seed is not None, the class (or
        experience) order will be shuffled according to the seed. When None, the
        current PyTorch random number generator state will be used. Defaults to
        None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing reproducibility.
        Defaults to None.
    :param shuffle: If True, the class (or experience) order will be shuffled.
        Defaults to True.
    :param class_ids_from_zero_in_each_exp:If True, original class IDs
        will be remapped so that they will appear as having an ascending
        order. For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        class_ids_from_zero_from_first_exp is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on.
        This is very useful when drawing confusion matrices and when dealing
        with algorithms with dynamic head expansion. Defaults to False.
        Mutually exclusive with the ``class_ids_from_zero_in_each_exp``
        parameter., defaults to False
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dummy: If True, the scenario will be a reduced version for testing
        and debugging purposes, defaults to False
    :return: A properly initialized :class:`NCScenario` instance.
    """

    # DEBUG for faster loading of the dataset
    # TODO fix naming, don't use dummy dirs
    if dummy:
        clickme_train = ClickMeDataset(root=root, split="val")
        clickme_test = ClickMeDataset(root=root, split="test")
    # TODO fix naming, don't use val, use train!
    else:
        clickme_train = ClickMeDataset(root=root, split="train")
        clickme_test = ClickMeDataset(root=root, split="test")

    return nc_benchmark(
        train_dataset=clickme_train,  # type: ignore
        test_dataset=clickme_test,  # type: ignore
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
