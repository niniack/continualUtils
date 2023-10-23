import torch
from torch import nn

from continualUtils.models import CustomResNet50


def test_model_weight_init():
    """Test initialization of CustomResNet50"""
    # Constants
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CustomResNet50(num_classes=num_classes, device=device)

    # Check if model.model is an instance of torch.nn.Module
    assert isinstance(
        model.model, torch.nn.Module
    ), "model.model is not an instance of torch.nn.Module!"

    # Check Kaiming Initialization for the first module
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)

            # Check if weights' mean is close to 0
            assert (
                torch.abs(m.weight.mean()) < 1e-6
            ), f"Mean of weights for {m} not close to 0"

            # Check if weights' variance is close to 2 / fan_in
            assert (
                torch.abs(m.weight.var() - 2 / fan_in) < 1e-6
            ), f"Variance of weights for {m} not initialized correctly"

            if m.bias is not None:
                # Check if bias is initialized to 0
                assert (
                    torch.abs(m.bias.mean()) < 1e-6
                ), f"Bias of {m} not initialized to 0"

            # Exit loop after checking the first module
            break


def test_model_accuracy(pretrained_resnet18, img_tensor_list):
    """Test accuracy of PretrainedResNet18"""
    model = pretrained_resnet18

    imagenet_cat_ids = [281, 282, 283, 284, 285, 286, 287]
    expected_cat = torch.argmax(model.forward(img_tensor_list[0]))

    imagenet_cowboy_hat = [515]
    expected_person = torch.argmax(model.forward(img_tensor_list[1]))

    assert (
        expected_cat in imagenet_cat_ids
        and expected_person in imagenet_cowboy_hat
    )
