import torch
from avalanche.evaluation.metric_results import MetricValue
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.benchmarks.datasets.clickme import (
    HEATMAP_INDEX,
    TOKEN_INDEX,
)
from continualUtils.training.losses.harmonizer_loss import NeuralHarmonizerLoss

# from activations.cam import GradCAM


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
    """Harmonization plugin"""

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.harmonizer = NeuralHarmonizerLoss(weight)

    # # EXPERIMENTAL
    # # TODO: Is this where we should initialize CAM extractor?
    # def before_training_exp(self, strategy, *args, **kwargs):
    #     # Set the CAM for this training experience
    #     self.cam = GradCAM(model=strategy.model,
    #                        target_layer=self.target_layer)

    def before_backward(self, strategy, **kwargs):
        cloned_mb_x = strategy.mb_x.detach()

        if not cloned_mb_x.requires_grad:
            cloned_mb_x.requires_grad_(True)

        # Get the heatmaps and tokens
        strategy.mb_heatmap = strategy.mbatch[HEATMAP_INDEX]
        strategy.mb_tokens = strategy.mbatch[TOKEN_INDEX]

        # Uses the `__call__` method
        harmonizer_loss = self.harmonizer(
            cloned_mb_x,
            strategy.mb_y,
            strategy.mb_heatmap,
            strategy.model,
            strategy.mb_tokens,
        )

        # Add the harmonizer loss to the overall loss
        # DEBUG make sure it ends up being +=
        strategy.loss += harmonizer_loss
        # strategy.loss = harmonizer_loss

        if torch.is_tensor(harmonizer_loss):
            harmonizer_loss = harmonizer_loss.cpu().item()

        step = strategy.clock.total_iterations
        exp_counter = strategy.clock.train_exp_counter
        mname = "harmonizer_loss/" + "exp" + str(exp_counter)
        mval = MetricValue(self, mname, harmonizer_loss, step)
        strategy.evaluator.publish_metric_value(mval)

    # # TODO: is this relevant?
    # def after_training_exp(self, strategy, **kwargs):
    #     self.harmonizer.update()

    def before_eval_exp(self, strategy, *args, **kwargs):
        strategy.harmonizer_loss = 0

    def before_eval_forward(self, strategy, *args, **kwargs):
        cloned_mb_x = strategy.mb_x.detach()

        if not cloned_mb_x.requires_grad:
            cloned_mb_x.requires_grad_(True)

        # Compute output and heatmap
        with torch.enable_grad():
            # Get the heatmaps and tokens
            strategy.mb_heatmap = strategy.mbatch[HEATMAP_INDEX]
            strategy.mb_tokens = strategy.mbatch[TOKEN_INDEX]

            # Uses the `__call__` method
            harmonizer_loss = self.harmonizer(
                cloned_mb_x,
                strategy.mb_y,
                strategy.mb_heatmap,
                strategy.model,
                strategy.mb_tokens,
            )

        # Disable gradient on input
        cloned_mb_x.requires_grad_(False)
        cloned_mb_x.detach()

        # Set loss
        strategy.mb_harmonizer_loss = harmonizer_loss
        strategy.harmonizer_loss += harmonizer_loss

        if torch.is_tensor(harmonizer_loss):
            harmonizer_loss = harmonizer_loss.cpu().item()

    def after_eval_exp(self, strategy, *args, **kwargs):
        num_eval_samples = len(strategy.dataloader.dataset)
        strategy.harmonizer_loss /= num_eval_samples

    # Avalanche defines loss *just* before this callback
    def after_eval_iteration(self, strategy, *args, **kwargs):
        # DEBUG make sure it ends up being +=
        strategy.loss += strategy.mb_harmonizer_loss
        # strategy.loss = strategy.mb_harmonizer_loss
