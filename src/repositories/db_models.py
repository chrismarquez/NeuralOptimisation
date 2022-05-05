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
class NeuralConfig(DataModel):
    learning_rate: float
    batch_size: int
    network_size: int
    depth: int
    activation_fn: Activation

    def __iter__(self):
        return iter((self.learning_rate, self.batch_size, self.network_size, self.depth, self.activation_fn))

    @staticmethod
    def from_dict(document: Dict) -> NeuralConfig:
        return from_dict(NeuralConfig, document)

@dataclass
class NeuralProperties(DataModel):
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
    function: str
    neural_config: NeuralConfig
    neural_properties: NeuralProperties
    model_data: bytes
    optimisation_properties: Optional[OptimisationProperties] = None
    id: Optional[str] = None
    experiment_id: Optional[str] = None

    @staticmethod
    def from_dict(document: Dict) -> NeuralModel:
        opt_props = document.get("optimisation_properties", None)
        opt_props = OptimisationProperties.from_dict(opt_props) if opt_props is not None else None
        return NeuralModel(
            id=str(document["_id"]),
            function=document["function"],
            neural_config=NeuralConfig.from_dict(document["neural_config"]),
            neural_properties=NeuralProperties.from_dict(document["neural_properties"]),
            optimisation_properties=opt_props,
            model_data=document["model_data"],
            experiment_id=document.get("experiment_id", None)
        )
