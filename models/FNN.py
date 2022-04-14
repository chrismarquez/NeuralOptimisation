from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torch.nn import Sequential


class FNN(nn.Module):

    def __init__(self):
        super().__init__()
        layers = OrderedDict(self._get_layers())
        self.layers = Sequential(layers)

    @staticmethod
    def _get_layers() -> List[(str, nn.Module)]:
        nodes = 10
        layer_count = 3
        layers = []
        prev = 2
        for i in range(layer_count):
            curr = 2 ** (layer_count - i) * nodes
            layers.append((f"linear{i + 1}", nn.Linear(in_features=prev, out_features=curr)))
            layers.append((f"relu{i + 1}", nn.ReLU()))
            prev = curr
        return layers

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


if __name__ == '__main__':
    net = FNN()
    print(net)
