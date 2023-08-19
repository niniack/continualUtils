import os
import torch
from pathlib import Path
from functools import reduce
from torch import nn
from transformers import ResNetConfig, ResNetModel, ResNetForImageClassification

import base
from avalanche.models import MultiHeadClassifier, MultiTaskModule


class CustomResNet50(base.BaseModel):
    def __init__(
        self, num_classes, device, seed=42, output_hidden=False, multihead=False
    ):
        """
        Returns:
            Resnet50 model
        """
        super().__init__(
            seed=seed,
            output_hidden=output_hidden,
            is_multihead=multihead,
            device=device,
            in_features=2048,
            out_features=num_classes,
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
            num_labels=num_classes,
        )

        self._model = ResNetForImageClassification(configuration).to(device)

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
            print("The provided model does not have a 'save_pretrained' method.")

    def _load_weights_impl(self, dir_name):
        print(f"Loading from {dir_name}")
        # Load model
        self._model = self.model.from_pretrained(dir_name)
        self._model = self.model.to(self.device)

    def forward(self, x, task_labels=None):
        if self.is_multihead:
            out = self.model.resnet(
                x, output_hidden_states=self.output_hidden, return_dict=True
            )
            # For multihead situation, must provide task labels!
            assert (
                task_labels != None
            ), "Failed to provide task labels for multihead classifier"

            # Reshape pooler output
            flat_pooler_out = out.pooler_output.view(out.pooler_output.size(0), -1)

            # Feed to multihead classifier
            classifier_out = self.multihead_classifier(flat_pooler_out, task_labels)
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
