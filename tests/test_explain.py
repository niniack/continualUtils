import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from continualUtils.benchmarks.datasets.clickme import HEATMAP_INDEX
from continualUtils.explain.tools import compute_saliency_map
from continualUtils.explain.tools.harmonizer_loss import NeuralHarmonizerLoss
from continualUtils.explain.tools.pyramidal import (
    _pyramidal_representation,
    compute_pyramidal_mse,
    standardize_cut,
)


def test_harmonizer_loss(pretrained_resnet18, split_clickme_benchmark):
    # Set up benchmark
    train_stream = split_clickme_benchmark.train_stream
    exp_set = train_stream[0].dataset
    (image, label, heatmap, token, task) = exp_set[0]

    # Define model
    model = pretrained_resnet18

    label = torch.tensor(label).long()
    input_img = image.unsqueeze(0).requires_grad_(True)
    heatmap = heatmap.unsqueeze(0)
    token = token.unsqueeze(0)
    output = model(input_img)

    sa_map = compute_saliency_map(output, input_img, label)
    sa_map_preprocess = standardize_cut(sa_map)
    heatmap_preprocess = standardize_cut(heatmap)

    # Get max
    eps = 1e-6
    with torch.no_grad():
        _sa_max = (
            torch.amax(sa_map_preprocess.detach(), dim=(2, 3), keepdim=True)
            + eps
        )
        _hm_max = torch.amax(heatmap_preprocess, dim=(2, 3), keepdim=True) + eps

        # Normalize the true heatmaps according to the saliency maps
        heatmap_preprocess = heatmap_preprocess / _hm_max * _sa_max

    manual_loss = compute_pyramidal_mse(
        sa_map_preprocess, heatmap_preprocess, token
    )

    nh_loss = NeuralHarmonizerLoss(weight=1)
    func_loss = nh_loss(
        mb_x=input_img,
        mb_y=label,
        mb_heatmap=heatmap,
        model=model,
        mb_tokens=token,
    )

    assert torch.allclose(manual_loss, func_loss)


def test_cut_ground_maps(split_clickme_benchmark):
    """Testing the output of standardizing SplitClickMe ground truth maps"""

    train_stream = split_clickme_benchmark.train_stream
    exp_set = train_stream[0].dataset
    ground_map = exp_set[0][HEATMAP_INDEX].unsqueeze(0)

    preprocessed_heatmaps = standardize_cut(ground_map)

    # This should only work pre-relu step
    # mean_val = torch.mean(preprocessed_heatmaps, dim=(2, 3)).item()
    # assert abs(mean_val) < 1e-5, f"Mean value is not close to 0, got {mean_val}"

    assert torch.min(preprocessed_heatmaps) >= 0.0


def test_compute_saliency_map(pretrained_resnet18, img_tensor_list):
    """Testing saliency map"""
    model = pretrained_resnet18
    inputs = img_tensor_list[0].requires_grad_(True)
    outputs = model.forward(img_tensor_list[0])
    targets = torch.Tensor([281]).long()

    # Compute the saliency map
    saliency_map = compute_saliency_map(outputs, inputs, targets)
    saliency_map_np = (
        saliency_map.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    )

    # Normalize the saliency map
    saliency_map_np = (saliency_map_np - saliency_map_np.min()) / (
        saliency_map_np.max() - saliency_map_np.min()
    )

    # Assert the shape
    assert inputs.shape[-2:] == saliency_map_np.shape[:2]


def test_pyramidal_mse():
    """
    Test the compute_pyramidal_mse function
    """
    heatmaps = torch.rand((10, 1, 224, 224))
    predicted_heatmaps = torch.rand((10, 1, 224, 224))
    tokens = torch.rand((10,))

    loss = compute_pyramidal_mse(heatmaps, predicted_heatmaps, tokens)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Checking for scalar tensor

    loss_none = compute_pyramidal_mse(
        heatmaps, predicted_heatmaps, torch.zeros((10,))
    )
    assert torch.allclose(loss_none, torch.zeros(1))

    loss_all = compute_pyramidal_mse(
        heatmaps, predicted_heatmaps, torch.ones((10,))
    )

    assert loss_all.item() >= loss.item() >= loss_none.item()


def test_pyramidal_representation():
    """Test the pyramidal representation"""
    # Define a sample maps tensor
    maps = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ]
    )

    # Apply _pyramidal_representation
    pyramid = _pyramidal_representation(maps, num_levels=2)

    # Check the sizes of the pyramid levels
    assert pyramid[0].shape == (1, 1, 4, 4)
    assert pyramid[1].shape == (1, 1, 2, 2)
    assert pyramid[2].shape == (1, 1, 1, 1)

    # Check some known values (you can expand on this)
    assert torch.allclose(
        pyramid[1][0, 0, 0, 0], torch.tensor(3.5)
    )  # Average of top-left 2x2 block
