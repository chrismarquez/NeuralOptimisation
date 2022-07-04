from typing import Tuple, List

import torch
from torch import nn

from models.LoadableModule import LoadableModule, Activation


class CNN(LoadableModule):

    def dummy_input(self) -> torch.Tensor:
        return torch.empty((1, 2))

    def __init__(self, depth: int, activation: Activation):
        super().__init__(activation)
        self.depth = depth

    def _get_layers(self) -> List[Tuple[str, nn.Module]]:
        activation_fn = self.get_activation()
        layers = []
        for i in range(self.depth):
            layers.append((f"conv2d{i + 1}", nn.Conv2d()))
            pass
        return layers


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass
