from typing import List

from experiments.Executor import ExperimentExecutor
from cluster.Job import Job
from optimisation.OptimisationJob import OptimisationJob
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.db_models import Bounds


class OptimisationExecutor(ExperimentExecutor):

    def __init__(self, repository: NeuralModelRepository):
        super().__init__()
        self._neural_repo = repository

    def _get_jobs(self) -> List[Job]:
        bounds = Bounds(0.2)
        id_list = self._neural_repo.get_all_id(non_optimised=True)
        return [OptimisationJob(model_id, bounds) for model_id in id_list]


if __name__ == '__main__':
    repo = NeuralModelRepository(uri="mongodb://localhost:27017")
    executor = OptimisationExecutor(repo)
    executor.run_all_jobs(use_cluster=False, test_run=True)
