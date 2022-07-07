"""
    Originally suggested in:
        https://discuss.pytorch.org/t/what-is-reshape-layer-in-pytorch/1110
"""

from typing import List, Tuple

import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args: int):
        super(Reshape, self).__init__()
        self.shape: Tuple[int, ...] = args

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print(input.size())
        x = input.view((input.size()[0], *self.shape))
        print(x.size())
        return x
