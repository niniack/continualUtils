from typing import Any

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.benchmarks.datasets.clickme import (
    HEATMAP_INDEX,
    TOKEN_INDEX,
)
from continualUtils.explain.tools.harmonizer_loss import NeuralHarmonizerLoss

# This is the logic/hierarchy for the strategy
# train
#     before_training

#     before_train_dataset_adaptation
#     train_dataset_adaptation
#     after_train_dataset_adaptation
#     make_train_dataloader
#     model_adaptation
#     make_optimizer
#     before_training_exp  # for each exp
#         before_training_epoch  # for each epoch
#             before_training_iteration  # for each iteration
#                 before_forward
#                 after_forward
#                 before_backward
#                 after_backward
#             after_training_iteration
#             before_update
#             after_update
#         after_training_epoch
#     after_training_exp
#     after_training

# eval
#     before_eval
#     before_eval_dataset_adaptation
#     eval_dataset_adaptation
#     after_eval_dataset_adaptation
#     make_eval_dataloader
#     model_adaptation
#     before_eval_exp  # for each exp
#         eval_epoch  # we have a single epoch in evaluation mode
#             before_eval_iteration  # for each iteration
#                 before_eval_forward
#                 after_eval_forward
#             after_eval_iteration
#     after_eval_exp
#     after_eval


class NeuralHarmonizerPlugin(SupervisedPlugin):
    """Neural Harmonizer plugin"""

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.harmonizer = NeuralHarmonizerLoss(weight)

    def before_forward(self, strategy: Any, *args, **kwargs):
        # Clean up harmonizer loss
        strategy.harmonizer_loss = torch.zeros(1, device=strategy.device)

    def before_backward(self, strategy: Any, *args, **kwargs):
        # Clone the input
        cloned_mb_x = strategy.mb_x.detach()

        # Turn on gradients for the cloned input
        if cloned_mb_x.requires_grad is False:
            cloned_mb_x.requires_grad_(True)

        # Get the heatmaps and tokens
        strategy.mb_heatmaps = strategy.mbatch[HEATMAP_INDEX]
        strategy.mb_tokens = strategy.mbatch[TOKEN_INDEX]
        strategy.mb_tasks = strategy.mbatch[4]

        # NH is a heavy operation, should only be done for samples with heatmaps
        # Compute all indices with token set to 1
        indices = torch.nonzero(strategy.mb_tokens, as_tuple=True)

        # Uses the `__call__` method
        # Send in the cloned input with gradients turned on
        # Select samples with heatmap
        harmonizer_loss = self.harmonizer(
            cloned_mb_x[indices],
            strategy.mb_y[indices],
            strategy.mb_heatmaps[indices],
            strategy.model[indices],
            strategy.mb_tokens[indices],
            strategy.mb_tasks[indices],
        )

        # Add the harmonizer loss
        strategy.loss += harmonizer_loss
        strategy.harmonizer_loss = harmonizer_loss

        # Turn off gradients for the cloned input
        cloned_mb_x.requires_grad_(False)
        cloned_mb_x.detach()

    # def before_eval_exp(self, strategy, *args, **kwargs):
    #     strategy.harmonizer_loss = 0

    # def before_eval_forward(self, strategy, *args, **kwargs):
    #     cloned_mb_x = strategy.mb_x.detach()

    #     if not cloned_mb_x.requires_grad:
    #         cloned_mb_x.requires_grad_(True)

    #     # Compute output and heatmap
    #     with torch.enable_grad():
    #         # Get the heatmaps and tokens
    #         strategy.mb_heatmap = strategy.mbatch[HEATMAP_INDEX]
    #         strategy.mb_tokens = strategy.mbatch[TOKEN_INDEX]

    #         # Uses the `__call__` method
    #         harmonizer_loss = self.harmonizer(
    #             cloned_mb_x,
    #             strategy.mb_y,
    #             strategy.mb_heatmap,
    #             strategy.model,
    #             strategy.mb_tokens,
    #         )

    #     # Disable gradient on input
    #     cloned_mb_x.requires_grad_(False)
    #     cloned_mb_x.detach()

    #     # Set loss
    #     strategy.mb_harmonizer_loss = harmonizer_loss
    #     strategy.harmonizer_loss += harmonizer_loss

    #     if torch.is_tensor(harmonizer_loss):
    #         harmonizer_loss = harmonizer_loss.cpu().item()

    # def after_eval_exp(self, strategy, *args, **kwargs):
    #     num_eval_samples = len(strategy.dataloader.dataset)
    #     strategy.harmonizer_loss /= num_eval_samples

    # # Avalanche defines loss *just* before this callback
    # def after_eval_iteration(self, strategy, *args, **kwargs):
    #     strategy.loss += strategy.mb_harmonizer_loss
