import gc
from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torch.nn import Sequential
from torchsummary import summary

from experiments.Polynomial import Polynomial
from models.LoadableModule import LoadableModule, Activation, Layer
from models.Reshape import Reshape


class CNN(LoadableModule):

    def dummy_input(self) -> torch.Tensor:
        return torch.empty((1, 2))

    def __init__(self, start_size: int, filter_size: int, filters: int, depth: int, activation: Activation):
        super().__init__(activation)
        self.start_size = start_size
        self.filter_size = filter_size
        self.filters = filters
        self.depth = depth

        layers = OrderedDict(self._get_layers())
        self.layers = Sequential(layers)

        gc.collect()
        torch.cuda.empty_cache()

    def _get_layers(self) -> List[Layer]:
        return self._get_initial_layer() + self._get_conv_layers() + self._get_fc_layer()

    def _get_initial_layer(self) -> List[Layer]:
        activation_fn = self.get_activation()
        return [
            (f"linear0", nn.Linear(in_features=2, out_features=self.start_size ** 2)),
            (f"act0", activation_fn()),
            (f"reshape", Reshape(1, self.start_size, self.start_size))
        ]

    def _get_conv_layers(self) -> List[Layer]:
        activation_fn = self.get_activation()
        in_channels = 1
        out_channels = self.filters
        layers = []
        for i in range(self.depth):
            layers.append((f"conv2d{i + 1}", nn.Conv2d(in_channels, out_channels, self.filter_size)))
            layers.append((f"act{i + 1}", activation_fn())),
            layers.append((f"pool{i + 1}", nn.MaxPool2d(2)))
            in_channels = out_channels
            out_channels = 2 * out_channels
        output_size = Polynomial.final_fc_coeff(self.start_size, self.filter_size, self.depth)
        layers.append(("flatten", Reshape(1, in_channels * output_size ** 2)))  # Flatten 3D Tensor to 1D Array
        return layers

    def _get_fc_layer(self) -> List[Layer]:
        output_size = Polynomial.final_fc_coeff(self.start_size, self.filter_size, self.depth)
        input_size = 2 ** (self.depth - 1) * self.filters * output_size ** 2
        activation_fn = self.get_activation()
        return [
            (f"linear{self.depth}", nn.Linear(input_size, output_size)),
            (f"act{self.depth}", activation_fn()),
            (f"out", nn.Linear(output_size, 1))
        ]


if __name__ == '__main__':
    net = CNN(start_size=20, filter_size=3, filters=20, depth=2, activation="ReLU")
    print(net)
    print(net.count_parameters())
    summary(net, (1, 2))
