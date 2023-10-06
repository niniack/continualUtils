import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin
from avalanche.training.templates import SupervisedTemplate
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from continualUtils.security.adversarial import (
    AdversarialReplayPlugin,
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


def test_avalanche_adversarial_plugin(av_simple_mlp, av_split_permuted_mnist):
    # loggers = []
    # loggers.append(InteractiveLogger())

    train_stream, test_stream = av_split_permuted_mnist

    # Prepare for training & testing
    optimizer = SGD(av_simple_mlp.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    # eval_plugin = EvaluationPlugin(
    #     accuracy_metrics(
    #         minibatch=True,
    #         epoch=True,
    #         epoch_running=True,
    #         experience=True,
    #         stream=True,
    #     ),
    #     # loggers=loggers,
    # )

    # Continual learning strategy
    ewc = EWCPlugin(ewc_lambda=0.001)
    adv_replay = AdversarialReplayPlugin()
    strategy = SupervisedTemplate(
        av_simple_mlp,
        optimizer,
        criterion,
        train_mb_size=2048,
        train_epochs=2,
        eval_mb_size=2048,
        plugins=[ewc, adv_replay],
    )

    # train and test loop
    results = []
    for train_task in train_stream:
        strategy.train(train_task, num_workers=2)
        results.append(strategy.eval(test_stream))
