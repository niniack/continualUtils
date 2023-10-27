import torch
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
        # Forward pass
        mb_pred = model(mb_x, mb_tasks)

        # Generate a saliency map
        sa_maps = compute_saliency_map(
            inputs=mb_x, targets=mb_y, outputs=mb_pred
        )

        # Unsqueeze to obtain channels
        heatmaps_preprocess = mb_heatmap

        # EXPERIMENTAL
        # Interpolation of SA map
        # mb_heatmap_dims = mb_heatmap.shape[-2:]
        # sa_maps_preprocess = torch.nn.functional.interpolate(
        #     sa_maps.unsqueeze(1), size=mb_heatmap_dims)

        # Standardize cut procedure
        sa_maps_preprocess = standardize_cut(sa_maps)
        heatmaps_preprocess = standardize_cut(heatmaps_preprocess)

        # Get max
        with torch.no_grad():
            _sa_max = (
                torch.amax(
                    sa_maps_preprocess.detach(), dim=(2, 3), keepdim=True
                )
                + self.epsilon
            )
            _hm_max = (
                torch.amax(heatmaps_preprocess, dim=(2, 3), keepdim=True)
                + self.epsilon
            )

            # Normalize the true heatmaps according to the saliency maps
            heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max

        # Pyramidal loss
        pyramidal_loss = compute_pyramidal_mse(
            sa_maps_preprocess, heatmaps_preprocess, mb_tokens
        )

        return self.weight * pyramidal_loss

    def update(self, *args, **kwargs):
        pass
