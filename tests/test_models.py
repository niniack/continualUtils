import torch
import torch.nn.functional as F
from torch import Tensor, nn

from continualUtils.models import CustomResNet50, PretrainedResNet50


def test_save_load(device, tmpdir):
    model = PretrainedResNet50(
        device=device,
        num_classes_per_head=10,
        multihead=True,
    )

    model.save_weights(f"{tmpdir}/model")

    model.load_weights(f"{tmpdir}/model")


def test_multihead(device, split_tiny_imagenet):
    """Tests multihead implementation"""
    model = CustomResNet50(
        device=device,
        num_classes_total=100,
        num_classes_per_head=10,
        multihead=True,
    )

    train_stream = split_tiny_imagenet.train_stream
    exp_set = train_stream[0].dataset
    image, *_ = exp_set[0]
    image = F.interpolate(image.unsqueeze(0), (224, 224)).to(device)

    model.adapt_model(experiences=train_stream[0])
    output = model(image)

    assert isinstance(output, dict)

    output = model(image, 0)

    assert isinstance(output, Tensor)


def test_patch_batch_norm(device):
    """Test initialization of CustomResNet50"""
    # Constants
    num_classes = 10

    # Initialize model
    model = CustomResNet50(
        device=device,
        num_classes_total=num_classes,
        patch_batch_norm=False,
    )

    # Check if model.model is an instance of torch.nn.Module
    assert isinstance(
        model.model, torch.nn.Module
    ), "model.model is not an instance of torch.nn.Module!"

    # Check for batch normalization layers using running statistics
    for layer in model.model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            assert (
                layer.track_running_stats
            ), "BatchNorm layer does not use running statistics!"


def test_model_weight_init(device):
    """Test initialization of CustomResNet50"""
    # Constants
    num_classes = 10
    epsilon = 1e-3

    # Initialize model
    model = CustomResNet50(
        device=device,
        num_classes_total=num_classes,
    )

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
                torch.abs(m.weight.mean()) < epsilon
            ), f"Mean of weights for {m} not close to 0"

            # Check if weights' variance is close to 2 / fan_in
            assert (
                torch.abs(m.weight.var() - 2 / fan_in) < epsilon
            ), f"Variance of weights for {m} not initialized correctly"

            if m.bias is not None:
                # Check if bias is initialized to 0
                assert (
                    torch.abs(m.bias.mean()) < epsilon
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
