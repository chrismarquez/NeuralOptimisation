import math
import os
from typing import List

import numpy as np

from src.cluster.Executor import Executor
from src.cluster.Job import Job
from src.data.Dataset import Dataset
from src.models.ModelJob import ModelJob
from src.repositories.NeuralModelRepository import NeuralModelRepository


class ModelsExecutor(Executor):

    def __init__(self, repository: NeuralModelRepository):
        super().__init__()
        self._repository = repository

    def _get_jobs(self) -> List[Job]:
        sizes_2 = [int(math.sqrt(2 * 10_000 * i + 15) - 4) for i in range(1, 10)]
        sizes_4 = [int(math.sqrt(32.0 / 21.0 * 10_000 * i - 32.0 / 21.0 + (64.0 / 21.0) ** 2) - 64.0 / 21.0) for i in
                   range(1, 9)]

        sizes_2 = [(it, 2) for it in sizes_2]
        sizes_4 = [(it, 4) for it in sizes_4]

        hyper_params = {
            "learning_rate": [1E-6, 3E-7],  # Evenly spaced lr in log scale
            "batch_size": [128, 512],
            "network_shape": sizes_2 + sizes_4,
            "activation_fn": ["ReLU", "Sigmoid"],
        }

        jobs = []
        for file in os.listdir("../resources/samples/"):
            raw_dataset = np.loadtxt(f"../resources/samples/{file}", delimiter=",")
            dataset = Dataset.create(raw_dataset)

            name, ext = file.split(".")

            job = ModelJob(name, dataset, hyper_params, self._repository)
            jobs.append(job)

            print(f"Computing params of function: {name}")
        return jobs


if __name__ == '__main__':
    repo = NeuralModelRepository()
    executor = ModelsExecutor(repo)
    executor.run_all_jobs()
