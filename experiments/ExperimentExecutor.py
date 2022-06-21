import asyncio
from typing import List, Awaitable, Optional

from cluster.Cluster import Cluster
from cluster.Job import Job
from cluster.JobContainer import JobContainer
from cluster.Pipeline import Segment, Pipeline
from experiments.Experiment import Experiment
from models.GridSearch import GridSearch
from models.ModelJob import ModelJob
from optimisation.OptimisationJob import OptimisationJob
from repositories.SampleDatasetRepository import SampleDatasetRepository
from repositories.db_models import Bounds


class ExperimentExecutor:

    @staticmethod
    def _run_locally(jobs: List[Job]):
        container = JobContainer()
        container.init_resources()
        container.wire(modules=[__name__])
        for job in jobs:
            print(job.encode())
            job.run(container)

    @staticmethod
    async def _optimisation_pipe(model_task: Awaitable[str]) -> Optional[Job]:
        model_id = await model_task
        bounds = Bounds(0.2)
        return OptimisationJob(model_id, bounds)

    def __init__(self, cluster: Cluster, sample_repo: SampleDatasetRepository):
        self._cluster = cluster
        self._sample_repo = sample_repo

        self._pipeline = Pipeline([
            Segment(
                name="Neural Trainer",
                capacity=50,
                submit=self._cluster.submit,
                pipe=ExperimentExecutor._optimisation_pipe
            ),
            Segment(
                name="Optimiser",
                capacity=200,
                submit=self._cluster.submit
            )
        ])

    async def run_experiment(
        self,
        experiment: Experiment,
        use_cluster: bool = True,
        test_run: bool = False
    ):
        jobs = self._get_initial_jobs(experiment)
        if test_run:
            jobs = [jobs[0]]
        if use_cluster:
            await self._pipeline.execute(jobs)
        else:
            ExperimentExecutor._run_locally(jobs)

    def _get_initial_jobs(self, experiment: Experiment) -> List[ModelJob]:
        hyper_params = experiment.hyper_params
        searcher = GridSearch()
        jobs = []
        for dataset_id in self._sample_repo.get_all_dataset_id():
            config_pool = searcher.get_sequence(hyper_params)
            for config in config_pool:
                job = ModelJob(dataset_id, config, experiment.exp_id)
                jobs.append(job)
        return jobs


if __name__ == '__main__':

    async def main():
        import os
        from repositories.SampleDatasetRepository import SampleDatasetRepository
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/experiments")[0]
        sample = SampleDatasetRepository("mongodb://localhost:27017/")
        cluster = Cluster(ROOT_DIR)
        experiment = Experiment("test-synchro")
        exec = ExperimentExecutor(cluster, sample)
        await exec.run_experiment(experiment, test_run=True, use_cluster=True)

    asyncio.run(main())
