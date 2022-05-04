from __future__ import annotations

from typing import List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.repositories.db_models import NeuralModel, NeuralProperties, OptimisationProperties


class NeuralModelRepository:

    def __init__(self):
        self._client = MongoClient()
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
            "id": model.id,
            "neural_properties": vars(model.neural_properties),
            "optimisation_properties": vars(model.optimisation_properties),
            "model_data": model.model_data
        }
        self._collection.insert_one(document)


if __name__ == '__main__':
    neural = NeuralProperties(1E-4, 32, 10, 3, "ReLU", 0.54, 0.95)
    opt = OptimisationProperties(0.0, 0.0, 0.5, 0.2, 12.3)
    model = NeuralModel(neural, opt, b"test")
    x = model.to_dict()
    repo = NeuralModelRepository()
    models = repo.get_all()
    print(model)
    print(models)
