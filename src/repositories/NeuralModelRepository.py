from __future__ import annotations

from typing import List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from tqdm import tqdm

from src.repositories.db_models import NeuralModel, NeuralProperties, OptimisationProperties
from src.views.Plot import Plot


class NeuralModelRepository:

    def __init__(self):
        self._client = MongoClient("mongodb://cloud-vm-42-88.doc.ic.ac.uk:27017")
        self._db: Database = self._client.NeuralOptimisation
        self._collection: Collection = self._db.NeuralModel

    def get(self, id: str) -> NeuralModel:
        document = self._collection.find_one({"_id": ObjectId(id)})
        return NeuralModel.from_dict(document)

    def get_all(self, experiment_id: Optional[str] = None) -> List[NeuralModel]:
        query = {} if experiment_id is None else {"experiment_id": experiment_id}
        documents = self._collection.find(query)
        return [NeuralModel.from_dict(document) for document in documents]

    def save(self, model: NeuralModel) -> None:
        document = {
            "neural_properties": vars(model.neural_properties),
            "optimisation_properties": vars(model.optimisation_properties),
            "model_data": model.model_data
        }
        self._collection.insert_one(document)


if __name__ == '__main__':
    repo = NeuralModelRepository()
    for function in tqdm(["ackley", "rastrigin", "rosenbrock", "sum_squares"], colour="green"):
        df = Plot.load_data(function)
        for id, row in tqdm(df.iterrows(), total=df.shape[0], colour="orange"):
            learning_rate, batch_size, network_size, depth, activation_fn, rmse, r2, x, y, location_error, optimum_error, computation_time = row
            neural_props = NeuralProperties(learning_rate, batch_size, network_size, depth, activation_fn, rmse, r2)
            optimisation_props = OptimisationProperties(x, y, location_error, optimum_error, computation_time)
            model_file = open(f"../../resources/trained/{function}/{id}.pt", "rb")
            model_data = model_file.read()
            model_file.close()
            model = NeuralModel(neural_props, optimisation_props, model_data, experiment_id="0")
            repo.save(model)
