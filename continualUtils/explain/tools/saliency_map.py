from typing import List, Union

import torch
import torch.nn.functional as F


def compute_saliency_map(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 1000,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Compute saliency map for given outputs, inputs, and targets.

    :param outputs: Model outputs.
    :param inputs: Model inputs.
    :param targets: Ground truth labels.
    :param create_graph: Whether to create a computation graph.
    :return: Computed saliency map.
    """

    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes)

    # Reduce outputs by summing over the product of outputs and one-hot targets
    outputs_reduced = torch.sum(outputs * targets_one_hot)

    # Compute gradients of reduced outputs with respect to inputs
    grads = torch.autograd.grad(
        outputs=outputs_reduced, inputs=inputs, create_graph=create_graph
    )[0]

    # Compute weights and saliency maps
    weights = grads.mean(dim=(2, 3), keepdim=True)
    saliency_maps = F.relu((grads * weights).sum(dim=1, keepdim=True))

    return saliency_maps
