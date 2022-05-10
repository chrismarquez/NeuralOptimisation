from typing import List

from src.cluster.Executor import Executor
from src.cluster.Job import Job
from src.optimisation.OptimisationJob import OptimisationJob
from src.repositories.NeuralModelRepository import NeuralModelRepository
from src.repositories.db_models import Bounds


class OptimisationExecutor(Executor):

    def __init__(self, repository: NeuralModelRepository):
        super().__init__()
        self._neural_repo = repository

    def _get_jobs(self) -> List[Job]:
        self._neural_repo.get_all()
        bounds = Bounds(0.2)
        id_list = self._neural_repo.get_all_id(non_optimised=True)
        return [OptimisationJob(self._neural_repo, model_id, bounds) for model_id in id_list]


if __name__ == '__main__':
    repo = NeuralModelRepository(uri="mongodb://localhost:27017")
    executor = OptimisationExecutor(repo)
    executor.run_all_jobs()
