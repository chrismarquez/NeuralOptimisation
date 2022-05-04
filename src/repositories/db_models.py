from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

from typing import Optional, Dict

from src.models.FNN import Activation

from dacite import from_dict


@dataclass
class DataModel(ABC):

    def to_dict(self):
        return asdict(self)

    @staticmethod
    @abstractmethod
    def from_dict(document: Dict):
        pass


@dataclass
class NeuralProperties(DataModel):
    learning_rate: float
    batch_size: int
    network_size: int
    depth: int
    activation_fn: Activation
    rmse: float
    r2: float

    @staticmethod
    def from_dict(document: Dict) -> NeuralProperties:
        return from_dict(NeuralProperties, document)


@dataclass
class OptimisationProperties(DataModel):
    x: float
    y: float
    location_error: float
    optimum_error: float
    computation_time: float

    @staticmethod
    def from_dict(document: Dict) -> OptimisationProperties:
        return from_dict(OptimisationProperties, document)


@dataclass
class NeuralModel(DataModel):
    neural_properties: NeuralProperties
    optimisation_properties: OptimisationProperties
    model_data: bytes
    id: Optional[str] = None
    experiment_id: Optional[str] = None

    @staticmethod
    def from_dict(document: Dict) -> NeuralModel:
        return NeuralModel(
            id=str(document["_id"]),
            neural_properties=NeuralProperties.from_dict(document["neural_properties"]),
            optimisation_properties=OptimisationProperties.from_dict(document["optimisation_properties"]),
            model_data=document["model_data"],
            experiment_id=document.get("experiment_id", None)
        )
