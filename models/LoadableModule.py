from __future__ import annotations

from typing import Callable, TypeVar
import torch
from torch import nn

from abc import ABC, abstractmethod

T = TypeVar('T')
Getter = Callable[[], T]


class LoadableModule(nn.Module, ABC):

    @abstractmethod
    def dummy_input(self) -> torch.Tensor:
        pass

    @staticmethod
    def load(path: str, build_net: Callable[[], LoadableModule]) -> LoadableModule:
        net = build_net()
        net.load_state_dict(torch.load(path))
        net.eval()
        return net
