import os
from functools import reduce
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import ResNetConfig, ResNetForImageClassification

from continualUtils.models import BaseModel, MissingTasksException


class CustomResNet50(BaseModel):
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
            Resnet50 model
        """
        super().__init__(
            seed=seed,
            device=device,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=512,
            num_classes_total=num_classes_total,
            num_classes_per_head=num_classes_per_head,
            init_weights=True,
        )

        # Initializing a model (with random weights) from
        # the resnet-50 style configuration
        configuration = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            hidden_act="relu",
            downsample_in_first_stage=False,
        )

        self._model = ResNetForImageClassification(configuration).to(device)  # type: ignore

        self._hidden_layers = [
            "resnet.embedder",
            "resnet.encoder.stages.0",
            "resnet.encoder.stages.1",
            "resnet.encoder.stages.2",
            "resnet.encoder.stages.3",
        ]
        self._num_hidden = len(self.hidden_layers)

    @property
    def model(self) -> ResNetForImageClassification:
        return self._model  # type: ignore

    @property
    def hidden_layers(self) -> List:
        return self._hidden_layers

    @property
    def num_hidden(self) -> int:
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
        self._model = self.model.to(self.device)  # type: ignore

    def forward(
        self, x, task_labels=None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Overrides forward method

        :param x: NCHW input
        :param task_labels: task labels for multihead, defaults to None
        :return: classifier output
        """
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

    def get_hidden_layer(self, idx):
        name = self.hidden_layers[idx]
        layers = name.split(".")
        return reduce(getattr, layers, self.model)
