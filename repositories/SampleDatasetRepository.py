from __future__ import annotations

import os
from typing import List

import numpy as np
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from repositories.db_models import SampleDataset, Sample


class SampleDatasetRepository:

    def __init__(self, uri: str):
        self._client = MongoClient(uri)
        self._db: Database = self._client.NeuralOptimisation
        self._root_collection: Collection = self._db.SampleDataset
        self._sample_collection: Collection = self._db.Sample

    def get(self, id: str) -> SampleDataset:
        document = self._root_collection.find_one({"_id": ObjectId(id)})
        cursor_samples = self._sample_collection.find({"dataset_id": ObjectId(id)})
        samples = {
            "samples": list(cursor_samples)
        }
        document = document | samples
        return SampleDataset.from_dict(document)

    def get_all(self) -> List[SampleDataset]:
        documents = self._root_collection.find()
        id_list = [str(doc["_id"]) for doc in documents]
        return [self.get(id) for id in id_list]

    def save(self, model: SampleDataset) -> None:
        document = model.to_dict()
        del document["id"]
        dataset_doc = {
            "function": model.function
        }
        result = self._root_collection.insert_one(dataset_doc)
        sample_docs = [
            {"dataset_id": result.inserted_id} | sample
            for sample in document["samples"]
        ]
        self._sample_collection.insert_many(sample_docs)


if __name__ == '__main__':
    repo = SampleDatasetRepository(uri="mongodb://localhost:27017")
    for file in os.listdir("../resources/samples/"):
        raw_dataset = np.loadtxt(f"../../resources/samples/{file}", delimiter=",")
        name, ext = file.split(".")
        samples = [
            {
                "x": raw_dataset[i, 0],
                "y": raw_dataset[i, 1],
                "z": raw_dataset[i, 2]
            }
            for i in range(raw_dataset.shape[0])
        ]
        samples = [Sample.from_dict(sample) for sample in samples]
        sample_dataset = SampleDataset(name, samples)
        repo.save(sample_dataset)
        print(f"Saved {file}")
