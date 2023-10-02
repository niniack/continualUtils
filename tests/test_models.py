import pytest
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from continualUtils.models import PretrainedResNet18


@pytest.fixture
def model():
    model = PretrainedResNet18(device=torch.device("cpu"))
    return model

@pytest.fixture
def images():
    # Define the necessary transformations: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Imagenet standards
    ])
    
    dataset = load_dataset("hf-internal-testing/fixtures_image_utils", split="test")
    
    image1 = Image.open(dataset[4]["file"]) # Cat image
    image2 = Image.open(dataset[5]["file"]) # Selena with hat image
    
    # Apply the transformations to the images and add a batch dimension
    tensor1 = transform(image1).unsqueeze(0)
    tensor2 = transform(image2).unsqueeze(0)

    images_tensor = [tensor1, tensor2]
    
    return images_tensor


def test_model_accuracy(model, images):

    imagenet_cat_ids = [281, 282, 283, 284, 285, 286, 287]
    expected_cat = torch.argmax(model.forward(images[0]))
    
    imagenet_cowboy_hat = [515]
    expected_person = torch.argmax(model.forward(images[1]))

    assert expected_cat in imagenet_cat_ids and expected_person in imagenet_cowboy_hat 