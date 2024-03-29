from __future__ import annotations

import gc
import tempfile
from typing import Callable, TypeVar, OrderedDict, Literal, Mapping, Tuple
import torch
from torch import nn

from abc import ABC, abstractmethod

from torch import Tensor

T = TypeVar('T')
Getter = Callable[[], T]

Activation = Literal["ReLU", "Tanh", "Sigmoid", "Softplus"]
ActivationFn = Callable[[], nn.Module]
Layer = Tuple[str, nn.Module]


class LoadableModule(nn.Module, ABC):
    _activations: Mapping[Activation, ActivationFn] = {
        "ReLU": lambda: nn.ReLU(),
        "Tanh": lambda: nn.Tanh(),
        "Sigmoid": lambda: nn.Sigmoid(),
        "Softplus": lambda: nn.Softplus()
    }

    @abstractmethod
    def dummy_input(self) -> torch.Tensor:
        pass

    def __init__(self, activation: Activation):
        super().__init__()
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

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

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def get_activation(self) -> ActivationFn:
        return LoadableModule._activations[self.activation]

    @staticmethod
    def _load_state_dict(file) -> OrderedDict[str, Tensor]:
        if torch.cuda.is_available():
            return torch.load(file)
        else:
            return torch.load(file, map_location=torch.device("cpu"))
