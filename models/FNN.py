from __future__ import annotations
from collections import OrderedDict
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential
from torchsummary import summary

from models.LoadableModule import LoadableModule


class FNN(LoadableModule):

    def dummy_input(self) -> torch.Tensor:
        return torch.empty((1, 2))

    @staticmethod
    def instantiate():
        return FNN(75, 2, activation_fn=lambda: nn.ReLU())

        # 20, 2 1s - fairly accurate
        # 50, 2 20s - fairly accurate
        # 50, 3 untractable

    @staticmethod
    def load(path: str, build_net: Callable[[], LoadableModule] = lambda: FNN.instantiate()) -> LoadableModule:
        return LoadableModule.load(path, build_net)

    def __init__(self, nodes: int, depth: int, activation_fn: Callable[[], nn.Module]):
        super().__init__()
        self.depth = depth
        self.nodes = nodes
        self.activation_fn = activation_fn

        layers = OrderedDict(self._get_layers())
        self.layers = Sequential(layers)

    def _get_layers(self) -> List[Tuple[str, nn.Module]]:
        layers = []
        prev = 2
        for i in range(self.depth):
            curr = 2 ** (self.depth - i) * self.nodes
            layers.append((f"linear{i + 1}", nn.Linear(in_features=prev, out_features=curr)))
            layers.append((f"act{i + 1}", self.activation_fn()))
            prev = curr
        layers.append((f"out", nn.Linear(in_features=prev, out_features=1)))
        return layers

    # [math.sqrt(0.25 * 10_000 * i +1) - 1 for i in range(1, 10)]
    # [(math.sqrt((1.0/32.0)* 10_000 * i + 1.0/9.0) - 1.0/3.0) / 3.0 for i in range(1, 9)]

    # [math.sqrt(2 * 10_000 * i + 15) - 4 for i in range(1, 10)]
    # [math.sqrt(32.0/21.0 * 10_000 * i - 32.0/21.0 + (64.0/21.0) ** 2) - 64.0/21.0 for i in range(1, 9)]


    def params(self):
        k = self.nodes
        d = self.depth
        return 2 * k * 2 ** d + np.sum([k ** 2 * 2 ** (2 * i + 1) for i in range(1, d)]) + \
               2 * k + 1 + np.sum([k * 2 ** i for i in range(1, d + 1)])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


if __name__ == '__main__':
    net = FNN.instantiate().cuda()
    print(net)
    print(net.params())
    summary(net, (128, 2))
