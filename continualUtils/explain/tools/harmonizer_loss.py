import torch
import torch.nn.functional as F
from avalanche.training.regularization import RegularizationMethod

from continualUtils.explain.tools import (
    compute_pyramidal_mse,
    compute_saliency_map,
    standardize_cut,
)


class NeuralHarmonizerLoss(RegularizationMethod):
    """Neural Harmonizer

    This method applies the neural harmonizer loss
    """

    def __init__(self, weight: float, epsilon: float = 1e-8):
        self.weight = weight
        self.epsilon = epsilon

    def __call__(self, mb_x, mb_y, mb_heatmap, model, mb_tokens, mb_tasks):
        # The input must have gradients turned on
        if not mb_x.requires_grad:
            mb_x.requires_grad_(True)

        # Generate a saliency map
        # Make targets one hot for our pure fn
        mb_y = F.one_hot(mb_y, model.num_classes)
        output_maps = compute_saliency_map(
            pure_function=compute_score,
            model=model,
            inputs=mb_x,
            tasks=mb_tasks,
            targets=mb_y,
        )

        # Standardize cut procedure
        output_maps_standardized = standardize_cut(output_maps)
        ground_maps_standardized = standardize_cut(mb_heatmap)

        # Normalize the true heatmaps according to the saliency maps
        # No gradients needed, we are just updating the ground truth maps
        with torch.no_grad():
            _om_max = (
                torch.amax(output_maps_standardized, dim=(2, 3), keepdim=True)
                + self.epsilon
            )
            _gm_max = (
                torch.amax(ground_maps_standardized, dim=(2, 3), keepdim=True)
                + self.epsilon
            )

            ground_maps_standardized = (
                ground_maps_standardized / _gm_max * _om_max
            )

        # Pyramidal loss
        pyramidal_loss = compute_pyramidal_mse(
            output_maps_standardized, ground_maps_standardized, mb_tokens
        )

        return pyramidal_loss

    def update(self, *args, **kwargs):
        pass


class OneHotException(Exception):
    pass


def compute_score(
    x: torch.Tensor,
    task: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """
    Since vmap will unbatch and vectorize the computation, we
    assume that all the inputs do not have a batch dimension.
    """

    # Batch
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    task = task.unsqueeze(0)

    # Forward pass with cloned input
    output = model(x, task)

    if output.shape != y.shape:
        raise OneHotException(
            "The model outputs must be the shape as the target."
        )
    score = torch.sum(output * y)
    return score
