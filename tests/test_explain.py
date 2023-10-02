import pytest
import torch

from continualUtils.explain import compute_saliency_map
from continualUtils.models import PretrainedResNet18


@pytest.fixture
def sample_data():
    model = PretrainedResNet18(num_classes=2, device=torch.device("cpu"))
    inputs = torch.randn((2, 3, 32, 32))  # Example inputs
    targets = torch.tensor([0, 1])  # Example targets

    return model, inputs, targets


def test_compute_saliency_map(sample_data):
    model, inputs, targets = sample_data

    print(dir(model))


    assert(False)

    # # Forward pass to get the outputs
    # outputs = model(inputs)

    # # Compute the saliency map
    # saliency_map = compute_saliency_map(outputs, inputs, targets)

    # # Perform some assertions
    # assert saliency_map is not None, "Saliency map should not be None"
    # assert saliency_map.shape == (
    #     2,
    #     1,
    #     32,
    #     32,
    # ), "Unexpected shape of saliency map"
    # assert torch.all(
    #     saliency_map >= 0
    # ), "Saliency map should have only non-negative values"
