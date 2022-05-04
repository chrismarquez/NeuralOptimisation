from __future__ import annotations

from typing import Optional, Dict

from src.models.FNN import Activation


class NeuralProperties:

    @staticmethod
    def from_dict(document: Dict) -> NeuralProperties:
        return NeuralProperties(
            learning_rate=document["learning_rate"],
            batch_size=document["batch_size"],
            network_size=document["network_size"],
            depth=document["depth"],
            activation_fn=document["activation_fn"],
            rmse=document["rmse"],
            r2=document["r2"],
        )

    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        network_size: int,
        depth: int,
        activation_fn: Activation,
        rmse: float,
        r2: float
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_size = network_size
        self.depth = depth
        self.activation_fn = activation_fn
        self.rmse = rmse
        self.r2 = r2


class OptimisationProperties:

    @staticmethod
    def from_dict(document: Dict) -> OptimisationProperties:
        return OptimisationProperties(
            x=document["x"],
            y=document["y"],
            location_error=document["location_error"],
            optimum_error=document["optimum_error"],
            computation_time=document["computation_time"],
        )

    def __init__(
        self,
        x: float,
        y: float,
        location_error: float,
        optimum_error: float,
        computation_time: float
    ):
        self.x = x
        self.y = y
        self.location_error = location_error
        self.optimum_error = optimum_error
        self.computation_time = computation_time


class NeuralModel:

    @staticmethod
    def from_dict(document: Dict) -> NeuralModel:
        return NeuralModel(
            id=document["_id"],
            neural_properties=NeuralProperties.from_dict(document["neuralProperties"]),
            optimisation_properties=OptimisationProperties.from_dict(document["optimisation_properties"]),
            model_data=document["model_data"],
            experiment_id=document.get("experiment_id", None)
        )

    def __init__(self, neural_properties: NeuralProperties, optimisation_properties: OptimisationProperties,
                 model_data: bytes, id: Optional[str] = None, experiment_id: Optional[str] = None):
        self.id = id
        self.neural_properties = neural_properties
        self.optimisation_properties = optimisation_properties
        self.model_data = model_data
        self.experiment_id = experiment_id
