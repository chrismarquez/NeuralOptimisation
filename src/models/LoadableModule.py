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

    def load(self, path: str) -> LoadableModule:
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
