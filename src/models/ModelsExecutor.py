import math
import os
from typing import List

import numpy as np

from src.cluster.Executor import Executor
from src.cluster.Job import Job
from src.data.Dataset import Dataset
from src.models.ModelJob import ModelJob


class ModelsExecutor(Executor):

    def __init__(self):
        super().__init__()

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
            "epochs": [200],
            "should_save": [True]
        }

        jobs = []
        for file in os.listdir("samples/"):
            raw_dataset = np.loadtxt(f"samples/{file}", delimiter=",")
            dataset = Dataset.create(raw_dataset)
            x_train, y_train = dataset.train

            name, ext = file.split(".")

            job = ModelJob(name, x_train, y_train, hyper_params)
            jobs.append(job)

            print(f"Computing params of function: {name}")
        return jobs


if __name__ == '__main__':
    executor = ModelsExecutor()
    executor.run_all_jobs()
