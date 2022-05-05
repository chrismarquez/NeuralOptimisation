import math
from typing import List

from src.cluster.Executor import Executor
from src.cluster.Job import Job
from src.models.ModelJob import ModelJob
from src.repositories.NeuralModelRepository import NeuralModelRepository
from src.repositories.SampleDatasetRepository import SampleDatasetRepository


class ModelsExecutor(Executor):

    def __init__(self, neural_repo: NeuralModelRepository, sample_repo: SampleDatasetRepository):
        super().__init__()
        self._neural_repo = neural_repo
        self._sample_repo = sample_repo

    def _get_jobs(self) -> List[Job]:
        hyper_params = ModelsExecutor._get_hyper_params()
        jobs = []
        for sample_dataset in self._sample_repo.get_all():
            function_name = sample_dataset.function
            dataset = sample_dataset.to_dataset()
            job = ModelJob(function_name, dataset, hyper_params, self._neural_repo)
            jobs.append(job)
            print(f"Computing params of function: {function_name}")
        return jobs

    @staticmethod
    def _get_hyper_params():
        sizes_2 = [int(math.sqrt(2 * 10_000 * i + 15) - 4) for i in range(1, 10)]
        sizes_4 = [int(math.sqrt(32.0 / 21.0 * 10_000 * i - 32.0 / 21.0 + (64.0 / 21.0) ** 2) - 64.0 / 21.0) for i in
                   range(1, 9)]

        sizes_2 = [(it, 2) for it in sizes_2]
        sizes_4 = [(it, 4) for it in sizes_4]

        return {
            "learning_rate": [1E-6, 3E-7],  # Evenly spaced lr in log scale
            "batch_size": [128, 512],
            "network_shape": sizes_2 + sizes_4,
            "activation_fn": ["ReLU", "Sigmoid"],
        }


if __name__ == '__main__':
    neural = NeuralModelRepository()
    sample = SampleDatasetRepository()
    executor = ModelsExecutor(neural, sample)
    executor.run_all_jobs()
