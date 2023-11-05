import os
from functools import reduce
from typing import Optional

import torch
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from torch import nn
from transformers import ResNetConfig, ResNetForImageClassification, ResNetModel

from continualUtils.models import BaseModel, MissingTasksException


class PretrainedResNet18(BaseModel):
    """Pretrained ResNet18 on Imagenet"""

    def __init__(
        self,
        device: torch.device,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
    ):
        _model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-18"
        ).to(device)

        super().__init__(
            seed=seed,
            device=device,
            model=_model,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=512,
            num_classes_total=1000,
            num_classes_per_head=num_classes_per_head,
            init_weights=False,
        )
        self._model = _model
        self._hidden_layers = []
        self._num_hidden = len(self.hidden_layers)

    @property
    def model(self):
        return self._model

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def num_hidden(self):
        return self._num_hidden

    def _save_weights_impl(self, dir_name):
        # Check if the model has a 'save_pretrained' method
        if hasattr(self.model, "save_pretrained"):
            # Create the directory if it doesn't exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # Save the model
            self.model.save_pretrained(dir_name)
            print(f"Model saved in directory: {dir_name}")
        else:
            print(
                "The provided model does not have a 'save_pretrained' method."
            )

    def _load_weights_impl(self, dir_name):
        print(f"Loading from {dir_name}")
        # Load model
        self._model = self.model.from_pretrained(dir_name)
        self._model = self.model.to(self.device)

    def get_hidden_layer(self, id):
        raise NotImplementedError("To Do!")

    def forward(self, x, task_labels=None):
        if self.is_multihead:
            if task_labels is None:
                raise MissingTasksException(
                    "Task labels must be provided for multihead classifiers"
                )

            out = self.model.resnet(
                x, output_hidden_states=self.output_hidden, return_dict=True
            )

            # Reshape pooler output
            flat_pooler_out = out.pooler_output.view(
                out.pooler_output.size(0), -1
            )

            # Feed to multihead classifier
            classifier_out = self.multihead_classifier(
                flat_pooler_out, task_labels
            )
        else:
            out = self.model(
                x, output_hidden_states=self.output_hidden, return_dict=True
            )
            classifier_out = out.logits

        if self.output_hidden:
            return classifier_out, out.hidden_states
        else:
            return classifier_out


class CustomResNet18(BaseModel):
    """Build a Resnet 18 model as
    described in https://arxiv.org/pdf/2007.07400.pdf
    """

    def __init__(
        self,
        device: torch.device,
        num_classes_total: int,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
    ):
        """
        Returns:
            Resnet18 model
        """

        # Initializing a model (with random weights) from
        # the resnet-50 style configuration
        configuration = ResNetConfig(
            num_channels=3,
            embedding_size=32,
            hidden_sizes=[32, 64, 128, 256],
            depths=[2, 2, 2, 2],
            layer_type="basic",
            hidden_act="relu",
            downsample_in_first_stage=True,
        )

        _model = ResNetForImageClassification(configuration).to(device)

        super().__init__(
            seed=seed,
            device=device,
            model=_model,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=512,
            num_classes_total=num_classes_total,
            num_classes_per_head=num_classes_per_head,
            init_weights=True,
        )

        self._model = _model
        self._hidden_layers = [
            "resnet.embedder",
            "resnet.encoder.stages.0",
            "resnet.encoder.stages.1",
            "resnet.encoder.stages.2",
            "resnet.encoder.stages.3",
        ]
        self._num_hidden = len(self.hidden_layers)

    @property
    def model(self):
        return self._model

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def num_hidden(self):
        return self._num_hidden

    def _save_weights_impl(self, dir_name):
        # Check if the model has a 'save_pretrained' method
        if hasattr(self.model, "save_pretrained"):
            # Create the directory if it doesn't exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # Save the model
            self.model.save_pretrained(dir_name)
            print(f"Model saved in directory: {dir_name}")
        else:
            print(
                "The provided model does not have a 'save_pretrained' method."
            )

    def _load_weights_impl(self, dir_name):
        print(f"Loading from {dir_name}")
        # Load model
        self._model = self.model.from_pretrained(dir_name)
        self._model = self.model.to(self.device)

    def forward(self, x, task_labels=None):
        if self.is_multihead:
            if task_labels is None:
                raise MissingTasksException(
                    "Task labels must be provided for multihead classifiers"
                )

            out = self.model.resnet(
                x, output_hidden_states=self.output_hidden, return_dict=True
            )

            # Reshape pooler output
            flat_pooler_out = out.pooler_output.view(
                out.pooler_output.size(0), -1
            )

            # Feed to multihead classifier
            classifier_out = self.multihead_classifier(
                flat_pooler_out, task_labels
            )
        else:
            out = self.model(
                x, output_hidden_states=self.output_hidden, return_dict=True
            )
            classifier_out = out.logits

        if self.output_hidden:
            return classifier_out, out.hidden_states
        else:
            return classifier_out

    def get_hidden_layer(self, id):
        name = self.hidden_layers[id]
        layers = name.split(".")
        return reduce(getattr, layers, self.model)
