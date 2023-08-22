import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from avalanche.training.regularization import RegularizationMethod

from typing import List, Union
from .loss_utils import _saliency_map, _standardize_cut, _pyramidal_mse

# from .loss_utils import _alt_saliency_map


class NeuralHarmonizerLoss(RegularizationMethod):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, mb_x, mb_y, mb_heatmap, model, mb_tokens):
        # Forward pass
        mb_pred = model(mb_x)

        # Generate a saliency map
        sa_maps = _saliency_map(inputs=mb_x, targets=mb_y, outputs=mb_pred)

        # Unsqueeze to obtain channels
        heatmaps_preprocess = mb_heatmap

        # EXPERIMENTAL
        # Interpolation of SA map
        # mb_heatmap_dims = mb_heatmap.shape[-2:]
        # sa_maps_preprocess = torch.nn.functional.interpolate(
        #     sa_maps.unsqueeze(1), size=mb_heatmap_dims)

        # Standardize cut procedure
        sa_maps_preprocess = _standardize_cut(sa_maps)
        heatmaps_preprocess = _standardize_cut(heatmaps_preprocess)

        # Get max
        with torch.no_grad():
            _sa_max = (
                torch.amax(sa_maps_preprocess.detach(), dim=(2, 3), keepdim=True) + 1e6
            )
            _hm_max = torch.amax(heatmaps_preprocess, dim=(2, 3), keepdims=True) + 1e6

            # Normalize the true heatmaps according to the saliency maps
            heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max

        # Pyramidal loss
        pyramidal_loss = _pyramidal_mse(
            sa_maps_preprocess, heatmaps_preprocess, mb_tokens
        )

        return self.weight * pyramidal_loss

    def update(self):
        return


def _saliency_map(outputs, inputs, targets, create_graph=True):
    # ex_batch_size = outputs.size()[0]
    # ex_index = outputs.argmax(dim=-1).view(-1, 1)
    # ex_one_hot = torch.zeros_like(outputs)
    # ex_one_hot.scatter_(-1, ex_index, 1.)
    # ex_out = torch.sum(ex_one_hot * outputs)

    # targets = targets.to(torch.long)
    # DEBUG!!!! chang eto 1000
    targets_one_hot = F.one_hot(targets, num_classes=1000)
    # targets_one_hot = F.one_hot(targets, num_classes=200)

    # ATTEMPT ONE
    # outputs_reduced = torch.sum(outputs * targets_one_hot, dim=-1)
    # grads = [torch.autograd.grad(outputs=out, inputs=mb_x, create_graph=create_graph)[
    # 0][i].unsqueeze(0) for i, out in enumerate(mb_pred_reduced)]
    # grads = torch.cat(grads, dim=0)

    # ATTEMPT TWO
    # outputs_reduced = torch.sum(outputs * targets_one_hot, dim=-1)
    # grads = torch.autograd.grad(outputs=outputs_reduced, inputs=inputs,
    #                             grad_outputs=torch.ones_like(outputs_reduced), create_graph=create_graph)[0]
    # grads = torch.mean(grads, dim=1, keepdim=True)

    # ATTEMPT THREE
    outputs_reduced = torch.sum(outputs * targets_one_hot)
    grads = torch.autograd.grad(
        outputs=outputs_reduced, inputs=inputs, create_graph=create_graph
    )[0]
    weights = grads.mean(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
    maps = F.relu((grads * weights).sum(dim=1))

    # unsqueeze in channel dim
    return maps.unsqueeze(1)


def _standardize_cut(heatmaps, axes=(2, 3), epsilon=1e-5):
    if heatmaps.dim() != 4:
        raise ValueError(
            f"Ensure that heatmaps are in NCHW cuDNN format, there are currently {heatmaps.dim()} dims"
        )

    means = torch.mean(heatmaps, dim=axes, keepdim=True)
    stds = torch.std(heatmaps, dim=axes, keepdim=True)

    heatmaps = heatmaps - means
    heatmaps = heatmaps / (stds + epsilon)

    heatmaps = F.relu(heatmaps)

    return heatmaps


def _pyramidal_mse(predicted_maps, true_maps, mb_tokens, num_levels=5):
    pyramid_y = _pyramidal_representation(true_maps, num_levels)
    pyramid_y_pred = _pyramidal_representation(predicted_maps, num_levels)

    pyramid_loss = [
        _mse(pyramid_y[i], pyramid_y_pred[i], mb_tokens) for i in range(num_levels)
    ]

    return torch.mean(torch.stack(pyramid_loss), dim=0)

    # # DEBUG
    # pyramid_loss = _mse(predicted_maps, true_maps, mb_tokens)
    # return torch.mean(pyramid_loss, dim=0)


def _mse(heatmaps_a, heatmaps_b, tokens):
    # return torch.mean(torch.square(heatmaps_a - heatmaps_b))
    return F.mse_loss(heatmaps_a, heatmaps_b)


def _pyramidal_representation(maps, num_levels):
    # # DEBUG changing downsampling
    # # Build a kernel
    # # maps have shape NCHW
    # kernel = _binomial_kernel(maps.shape[1])

    # # annoying way to bring filter to same device
    # kernel = kernel.cuda(maps.get_device()) if maps.is_cuda else kernel

    levels = [maps]
    for _ in range(num_levels):
        # # DEBUG changing downsampling
        # maps = _downsample(maps, kernel)
        new_size = maps.shape[-1] // 2
        maps = F.interpolate(maps, size=new_size, mode="bilinear")
        levels.append(maps)
    return levels


# # DEBUG changing downsampling
# def _binomial_kernel(num_channels):
#     kernel = torch.FloatTensor([1., 4., 6., 4., 1.])
#     kernel = torch.outer(kernel, kernel)
#     kernel /= torch.sum(kernel)
#     return kernel.repeat((num_channels, num_channels, 1, 1))

# # DEBUG changing downsampling
# def _downsample(maps, kernel):
#     return F.conv2d(input=maps, weight=kernel, stride=2, padding='valid')

# How to use
# Returns a list
# sa_maps = _alt_saliency(mb_x, mb_y, mb_pred)[0]

# def _alt_saliency_map(mb_x, mb_y, mb_pred, cam):
#     '''
#     Returns a list
#     '''
#     mb_pred_argmax = mb_pred.argmax(dim=1).tolist()
#     grads = cam(mb_pred_argmax, mb_pred, retain_graph=True)
#     return grads
