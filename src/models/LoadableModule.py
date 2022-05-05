from __future__ import annotations

import tempfile
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

    def load_bytes(self, buffer: bytes) -> LoadableModule:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
            file.write(buffer)
            file.seek(0)
            self.load_state_dict(torch.load(file))
            self.eval()
            return self
