from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Union

import numpy as np
from bson import ObjectId
from dacite import from_dict

from data.Dataset import Dataset
from experiments.Experiment import NeuralType
from models.FNN import Activation
# Bounds = Mapping[int, Tuple[float, float]]
from optimisation.Solver import Solver


@dataclass
class DataModel(ABC):

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    @abstractmethod
    def from_dict(document: Dict):
        pass


@dataclass
class Bounds(DataModel):
    range: float
    dim: int = 2

    def from_dict(self, document: Dict) -> Bounds:
        return from_dict(Bounds, document)

    def to_pyomo_bounds(self) -> Dict:
        return {
            i: (-self.range, self.range) for i in range(self.dim)
        }


@dataclass
class FeedforwardNeuralConfig(DataModel):
    learning_rate: float
    batch_size: int
    network_size: int
    depth: int
    activation_fn: Activation

    def __iter__(self):
        return iter((self.learning_rate, self.batch_size, self.network_size, self.depth, self.activation_fn))

    def get_neural_type(self) -> NeuralType:
        return "Feedforward"

    @staticmethod
    def from_dict(document: Dict) -> FeedforwardNeuralConfig:
        return from_dict(FeedforwardNeuralConfig, document)


@dataclass
class ConvolutionalNeuralConfig(DataModel):
    learning_rate: float
    batch_size: int
    start_size: int
    filters: int
    filter_size: int
    depth: int
    activation_fn: Activation

    def __iter__(self):
        return iter((self.learning_rate, self.batch_size, self.filters, self.filter_size, self.depth, self.activation_fn))

    def get_neural_type(self) -> NeuralType:
        return "Convolutional"

    @staticmethod
    def from_dict(document: Dict) -> ConvolutionalNeuralConfig:
        return from_dict(ConvolutionalNeuralConfig, document)


NeuralConfig = Union[FeedforwardNeuralConfig, ConvolutionalNeuralConfig]


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
    input_bounds: Bounds
    solver_type: Solver
    location_error: float
    optimum_error: float
    computation_time: float
    successful: bool

    @staticmethod
    def from_dict(document: Dict) -> OptimisationProperties:
        return from_dict(OptimisationProperties, document)


@dataclass
class NeuralModel(DataModel):
    function: str
    type: NeuralType
    neural_config: NeuralConfig
    expected_optimisations: int
    l1_reg_lambda: Optional[float]
    model_data: Optional[bytes] = None
    neural_properties: Optional[NeuralProperties] = None
    optimisation_properties: List[OptimisationProperties] = field(default_factory=list)
    id: Optional[str] = None
    experiment_id: Optional[str] = None

    def is_complete(self) -> bool:
        if self.model_data is None or self.neural_properties is None:
            return False
        if len(self.optimisation_properties) != self.expected_optimisations:
            return False
        return True

    @staticmethod
    def _get_neural_config(document: Dict) -> NeuralConfig:
        if document["type"] == "Feedforward":
            return FeedforwardNeuralConfig.from_dict(document["neural_config"])
        elif document["type"] == "Convolutional":
            return ConvolutionalNeuralConfig.from_dict(document["neural_config"])
        else:
            raise RuntimeError("Unknown Neural Config")

    @staticmethod
    def from_dict(document: Dict) -> NeuralModel:
        opt_props = document.get("optimisation_properties", None)
        opt_props = [OptimisationProperties.from_dict(props) for props in opt_props] if opt_props is not None else []
        neural_props = document.get("neural_properties", None)
        neural_props = NeuralProperties.from_dict(document["neural_properties"]) if neural_props is not None else None
        return NeuralModel(
            id=str(document["_id"]),
            function=document["function"],
            type=document["type"],
            neural_config=NeuralModel._get_neural_config(document),
            l1_reg_lambda=document["l1_reg_lambda"],
            expected_optimisations=document["expected_optimisations"],
            model_data=document.get("model_data", None),
            neural_properties=neural_props,
            optimisation_properties=opt_props,
            experiment_id=document.get("experiment_id", None)
        )


@dataclass
class SampleDataset(DataModel):
    function: str
    samples: List[Sample]
    id: Optional[str] = None

    @staticmethod
    def from_dict(document: Dict) -> SampleDataset:
        return SampleDataset(
            id=str(document["_id"]),
            function=document["function"],
            samples=[Sample.from_dict(sample) for sample in document["samples"]]
        )

    def to_dataset(self) -> Dataset:
        samples = [[sample.x, sample.y, sample.z] for sample in self.samples]
        return Dataset.create(np.array(samples))


@dataclass
class Sample(DataModel):
    x: float
    y: float
    z: float

    @staticmethod
    def from_dict(document: Dict) -> Sample:
        return Sample(document["x"], document["y"], document["z"])


if __name__ == '__main__':
    doc = {
        "_id": ObjectId("62742376fb25904035ab06c9"),
        "function": "ReLU",
        "samples": [
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3},
            {"x": 0.1, "y": 0.2, "z": 0.3}
        ]
    }
    dataset = SampleDataset.from_dict(doc)
    other_doc = dataset.to_dict()
    x = dataset.to_dataset()
    print(dataset.to_dataset())
    print(other_doc)
