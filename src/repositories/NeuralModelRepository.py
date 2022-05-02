
from pymongo import MongoClient

from src.models.FNN import Activation


class NeuralProperties:

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

    def __init__(self, id: str, neuralProperties: NeuralProperties, optimisationProperties: OptimisationProperties,
                 modelData: bytes):
        super().__init__()
        self.id = id
        self.neuralProperties = neuralProperties
        self.optimisationProperties = optimisationProperties
        self.modelData = modelData


class NeuralModelRepository:

    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.NeuralOptimisation
        self.collection = self.db.NeuralModel

    def save(self, model: NeuralModel):
        payload = {
            "id": model.id,
            "neuralProperties": vars(model.neuralProperties),
            "optimisationProperties": vars(model.optimisationProperties),
            "modelData": model.modelData
        }
        self.collection.insert_one(payload)


if __name__ == '__main__':
    neural = NeuralProperties(1E-4, 32, 10, 3, "ReLU", 0.54, 0.95)
    opt = OptimisationProperties(0.0, 0.0, 0.5, 0.2, 12.3)
    model = NeuralModel("test-ID", neural, opt, b"test")
    repo = NeuralModelRepository()
    repo.save(model)
