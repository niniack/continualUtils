import torch

from continualUtils.security.tools.universal_adversarial_perturbation import (
    UniversalAdversarialPerturbation,
)


def test_adversarial_uap(pretrained_resnet18, img_tensor_dataset):
    """Test universal adversarial perturbation"""
    # Define the attack
    attack = UniversalAdversarialPerturbation()
    eps = 0.3

    # Apply the attack
    transform = attack(
        img_tensor_dataset,
        pretrained_resnet18,
        learning_rate=0.01,
        iterations=10,
        eps=eps,
        device=torch.device("cpu"),
    )

    # Test the adversarial noise
    img, *_ = img_tensor_dataset[0]
    perturbed_img, *_ = transform(img_tensor_dataset)[0]
    noise = perturbed_img - img
    assert torch.all(torch.isclose(noise, torch.zeros_like(noise), atol=eps))
