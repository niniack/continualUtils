import matplotlib.pyplot as plt
import pytest
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from continualUtils.explain import compute_saliency_map
from continualUtils.models import PretrainedResNet18


def test_compute_saliency_map(pretrained_resnet18, img_tensor_list):
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

    # Compare with baseline
    cam_engine = GradCAM(
        model=model,
        target_layers=[model.model.resnet.encoder.stages[-1]],
        use_cuda=True,
    )
    cam = cam_engine(input_tensor=inputs, targets=[ClassifierOutputTarget(281)])

    # Display the saliency map
    plt.imshow(cam.transpose(1, 2, 0), cmap="hot")
    plt.axis("off")
    plt.colorbar()
    plt.savefig("test2", bbox_inches="tight")
    plt.show()

    # Assert the shape
    assert inputs.shape[-2:] == saliency_map_np.shape[:2]
