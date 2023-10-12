import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from avalanche.training.regularization import RegularizationMethod

from continualUtils.explain.tools import (
    compute_pyramidal_mse,
    compute_saliency_map,
    standardize_cut,
)

# from .loss_utils import _alt_saliency_map


class NeuralHarmonizerLoss(RegularizationMethod):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, mb_x, mb_y, mb_heatmap, model, mb_tokens):
        # Forward pass
        mb_pred = model(mb_x)

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
                + 1e6
            )
            _hm_max = (
                torch.amax(heatmaps_preprocess, dim=(2, 3), keepdim=True) + 1e6
            )

            # Normalize the true heatmaps according to the saliency maps
            heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max

        # Pyramidal loss
        pyramidal_loss = compute_pyramidal_mse(
            sa_maps_preprocess, heatmaps_preprocess, mb_tokens
        )

        return self.weight * pyramidal_loss

    def update(self):
        return
