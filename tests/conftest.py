import pytest
import torch
import torchvision.transforms as transforms
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from datasets import load_dataset
from PIL import Image
from torch.utils.data import TensorDataset

from continualUtils.models import PretrainedResNet18


@pytest.fixture
def pretrained_resnet18():
    model = PretrainedResNet18(device=torch.device("cpu"))
    return model


@pytest.fixture
def img_tensor_list():
    # Define the necessary transformations: convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Imagenet standards
        ]
    )

    dataset = load_dataset(
        "hf-internal-testing/fixtures_image_utils", split="test"
    )

    image1 = Image.open(dataset[4]["file"])  # Cat image
    image2 = Image.open(dataset[5]["file"])  # Selena with hat image

    # Apply the transformations to the images and add a batch dimension
    tensor1 = transform(image1).unsqueeze(0)
    tensor2 = transform(image2).unsqueeze(0)

    images_tensor = [tensor1, tensor2]

    return images_tensor


@pytest.fixture
def img_tensor_dataset():
    # Define the necessary transformations: convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Imagenet standards
        ]
    )

    dataset = load_dataset(
        "hf-internal-testing/fixtures_image_utils", split="test"
    )

    image1 = Image.open(dataset[4]["file"])  # Cat image

    # Apply the transformations to the images and add a batch dimension
    tensor1 = transform(image1).unsqueeze(0)

    return TensorDataset(tensor1, torch.Tensor([281]).long())


@pytest.fixture
def av_split_permuted_mnist():
    perm_mnist = PermutedMNIST(n_experiences=2)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream
    return train_stream, test_stream


@pytest.fixture
def av_simple_mlp():
    model = SimpleMLP(num_classes=10)
    return model
