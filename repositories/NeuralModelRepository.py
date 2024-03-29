from __future__ import annotations

from typing import List, Optional, Dict

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from tqdm import tqdm

from repositories.db_models import NeuralModel, NeuralProperties, OptimisationProperties, FeedforwardNeuralConfig, \
    Bounds


class NeuralModelRepository:

    def __init__(self, uri: str):
        print(f"[NeuralModelRepository] Connecting to DB at: {uri}")
        self._client = MongoClient(uri)
        self._db: Database = self._client.NeuralOptimisation
        self._collection: Collection = self._db.NeuralModel

    def get(self, id: str) -> NeuralModel:
        document: Optional[Dict] = self._collection.find_one({"_id": ObjectId(id)})
        if document is None:
            raise RuntimeError(f"Document with ID {id} not found")
        return NeuralModel.from_dict(document)

    def get_all_id(self, function: str = None, non_optimised: bool = False):
        optimised_query = {"optimisation_properties": None} if non_optimised else {}
        function_query = {"function": function} if function is not None else {}
        query = optimised_query | function_query
        id_list = self._collection.find(query).distinct("_id")
        return [str(model_id) for model_id in id_list]

    def count_total(self, exp_id) -> int:
        query = {"experiment_id": exp_id}
        return self._collection.count_documents(query)

    def experiment_exists(self, exp_id: str) -> bool:
        query = {"experiment_id": exp_id}
        return self._collection.count_documents(query) > 0

    def count_models_to_train(self, exp_id: str) -> int:
        query = {
            "neural_properties": None,
            "experiment_id": exp_id
        }
        return self._collection.count_documents(query)

    def get_all(self, experiment_id: Optional[str] = None) -> List[NeuralModel]:
        query = {} if experiment_id is None else {"experiment_id": experiment_id}
        documents = self._collection.find(query)
        return [NeuralModel.from_dict(document) for document in documents]

    def save(self, model: NeuralModel) -> str:
        document = model.to_dict()
        del document["id"]
        result = self._collection.insert_one(document)
        return str(result.inserted_id)

    def update(self, model: NeuralModel) -> None:
        query = {"_id": ObjectId(model.id)}
        document = model.to_dict()
        del document["id"]
        self._collection.replace_one(query, document)


if __name__ == '__main__':
    from views.Plot import Plot
    repo = NeuralModelRepository(uri="mongodb://localhost:27017")
    input_bounds = Bounds(0.2)
    for function in tqdm(["ackley", "rastrigin", "rosenbrock", "sum_squares"], colour="green"):
        df = Plot.load_data_from_file(function)
        for id, row in tqdm(df.iterrows(), total=df.shape[0], colour="orange"):
            learning_rate, batch_size, network_size, depth, activation_fn, rmse, r2, x, y, location_error, optimum_error, computation_time = row
            neural_config = FeedforwardNeuralConfig(learning_rate, batch_size, network_size, depth, activation_fn)
            neural_props = NeuralProperties(rmse, r2)
            optimisation_props = OptimisationProperties(
                x, y, input_bounds, location_error, optimum_error, computation_time, 0.0, True
            )
            model_file = open(f"../../resources/trained/{function}/{id}.pt", "rb")
            model_data = model_file.read()
            model_file.close()
            model = NeuralModel(
                function, "Feedforward", neural_config, 2, model_data, neural_props, [optimisation_props], experiment_id="0"
            )
            repo.save(model)
