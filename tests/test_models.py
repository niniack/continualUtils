import torch


def test_model_accuracy(pretrained_resnet18, img_tensor_list):
    model = pretrained_resnet18

    imagenet_cat_ids = [281, 282, 283, 284, 285, 286, 287]
    expected_cat = torch.argmax(model.forward(img_tensor_list[0]))

    imagenet_cowboy_hat = [515]
    expected_person = torch.argmax(model.forward(img_tensor_list[1]))

    assert (
        expected_cat in imagenet_cat_ids
        and expected_person in imagenet_cowboy_hat
    )
