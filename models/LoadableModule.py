from __future__ import annotations

import gc
import tempfile
from typing import Callable, TypeVar, OrderedDict
import torch
from torch import nn

from abc import ABC, abstractmethod

from torch import Tensor

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
            state_dict = LoadableModule._load_state_dict(file)
            self.load_state_dict(state_dict)
            self.eval()
            gc.collect()
            torch.cuda.empty_cache()
            return self

    @staticmethod
    def _load_state_dict(file) -> OrderedDict[str, Tensor]:
        if torch.cuda.is_available():
            return torch.load(file)
        else:
            return torch.load(file, map_location=torch.device("cpu"))
