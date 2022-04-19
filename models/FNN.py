from collections import OrderedDict
from typing import List, Tuple, Callable

import torch
from torch import nn
from torch.nn import Sequential
from torchsummary import summary


class FNN(nn.Module):

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


if __name__ == '__main__':
    net = FNN(50, 3, activation_fn=lambda: nn.ReLU()).cuda()
    print(net)
    summary(net, (128, 2))
