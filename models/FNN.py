from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple, Callable

import torch
from torch import nn
from torch.nn import Sequential

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
        self.depth = depth  # network depth
        self.nodes = nodes  # nodes of layer 1
        self.activation_fn = activation_fn

        layers = OrderedDict(self._get_layers())
        self.layers = Sequential(layers)

    def _get_layers(self) -> List[Tuple[str, nn.Module]]:
        hidden = [int(self.nodes / (2 ** i)) for i in range(self.depth)]
        sizes = [2] + hidden
        layers = []
        for i in range(self.depth):
            prev = sizes[i]
            curr = sizes[i + 1]
            layers.append((f"linear{i + 1}", nn.Linear(in_features=prev, out_features=curr)))
            layers.append((f"act{i + 1}", self.activation_fn()))
        layers.append((f"out", nn.Linear(in_features=sizes[-1], out_features=1)))
        return layers

    # [math.sqrt(0.25 * 10_000 * i + 1) - 1 for i in range(1, 10)]
    # [(math.sqrt((1.0/32.0)* 10_000 * i + 1.0/9.0) - 1.0/3.0) / 3.0 for i in range(1, 9)]

    # [math.sqrt(2 * 10_000 * i + 15) - 4 for i in range(1, 10)]
    # [math.sqrt(32.0/21.0 * 10_000 * i - 32.0/21.0 + (64.0/21.0) ** 2) - 64.0/21.0 for i in range(1, 9)]

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


if __name__ == '__main__':
    net = FNN(nodes=300, depth=4, activation_fn=lambda: nn.ReLU()).cuda()
    print(net)
    print(net.count_parameters())
    #summary(net, (128, 2))
