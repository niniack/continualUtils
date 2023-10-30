import inspect
from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import grad, vmap


def check_pure_function(func):
    """Checks whether pure function requires the arguments
    inputs, model, and targets

    :param func: pure function to check
    :return: _description_
    """
    signature = inspect.signature(func)
    parameters = list(signature.parameters.keys())

    required_args = {"x", "task", "y", "model"}
    return required_args.issubset(parameters)


def compute_saliency_map(
    pure_function: Callable,
    model: torch.Tensor,
    inputs: torch.Tensor,
    tasks: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute saliency map

    :param pure_function: Callable function.
    :param inputs: Model inputs.
    :param targets: Ground truth labels.
    :return: Computed saliency map.
    """

    # Check the pure function
    if not check_pure_function(pure_function):
        raise ValueError(
            f"{pure_function.__name__} must have the arguments: inputs, model, targets"
        )

    # Set up gradient operator with respect to first argument
    # The 0th argument MUST be inputs
    compute_single_saliency = grad(pure_function, argnums=0, has_aux=False)

    # Set up vmap operator for entire batch
    # All arguments must be batched (see in_dims)
    compute_batch_saliency = vmap(
        compute_single_saliency, in_dims=(0, 0, 0, None)
    )

    # Execute the transformed function
    # vmap will automatically unbatch the arguments
    per_sample_grad = compute_batch_saliency(inputs, tasks, targets, model)

    # Reduce the channels to get single channel heatmap
    per_sample_map = torch.mean(per_sample_grad, dim=1, keepdim=True)

    # ReLU on the heatmap
    per_sample_map = F.relu(per_sample_map)

    return per_sample_map


## Archive ##

# def compute_saliency_map(
#     outputs: torch.Tensor,
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     num_classes: int = 1000,
#     create_graph: bool = True,
# ) -> torch.Tensor:
#     """
#     Compute saliency map for given outputs, inputs, and targets.

#     :param outputs: Model outputs.
#     :param inputs: Model inputs.
#     :param targets: Ground truth labels.
#     :param create_graph: Whether to create a computation graph.
#     :return: Computed saliency map.
#     """

#     # Convert targets to one-hot encoding
#     targets_one_hot = F.one_hot(targets, num_classes)

#     # Reduce outputs by summing over the product of outputs and one-hot targets
#     outputs_reduced = torch.sum(outputs * targets_one_hot)

#     # Compute gradients of reduced outputs with respect to inputs
#     grads = torch.autograd.grad(
#         outputs=outputs_reduced, inputs=inputs, create_graph=create_graph
#     )[0]

#     # Compute weights and saliency maps
#     weights = grads.mean(dim=(2, 3), keepdim=True)
#     saliency_maps = F.relu((grads * weights).sum(dim=1, keepdim=True))

#     return saliency_maps
