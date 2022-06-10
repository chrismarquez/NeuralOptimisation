import math
from typing import List

from cluster.Executor import Executor
from cluster.Job import Job
from models.GridSearch import GridSearch
from models.ModelJob import ModelJob
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository


class ModelsExecutor(Executor):

    def __init__(self, neural_repo: NeuralModelRepository, sample_repo: SampleDatasetRepository):
        super().__init__()
        self._neural_repo = neural_repo
        self._sample_repo = sample_repo

    def _get_jobs(self) -> List[Job]:
        hyper_params = ModelsExecutor._get_hyper_params()
        searcher = GridSearch()
        jobs = []
        for dataset_id in self._sample_repo.get_all_dataset_id():
            config_pool = searcher.get_sequence(hyper_params)
            for config in config_pool:
                job = ModelJob(dataset_id, config, self._neural_repo, self._sample_repo)
                jobs.append(job)
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
