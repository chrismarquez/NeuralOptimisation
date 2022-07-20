from dataclasses import dataclass
from typing import List, Awaitable

from cluster.Cluster import Cluster
from cluster.Job import Job
from cluster.JobContainer import JobContainer
from cluster.Pipeline import Segment, Pipeline
from experiments.Experiment import Experiment
from models.GridSearch import GridSearch
from models.ModelJob import ModelJob
from optimisation.OptimisationJob import OptimisationJob
from optimisation.Solver import solvable_by
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository
from repositories.db_models import Bounds


@dataclass
class ExpectedJobs:
    training: int
    optimisation: int


class ExperimentExecutor:

    def __init__(self, cluster: Cluster, neural_repo: NeuralModelRepository, sample_repo: SampleDatasetRepository):
        self._cluster = cluster
        self._neural_repo = neural_repo
        self._sample_repo = sample_repo

    async def run_experiment(
        self,
        experiment: Experiment,
        use_cluster: bool = True,
        test_run: bool = False
    ):
        jobs = self._get_initial_jobs(experiment)
        if test_run:
            jobs = jobs[0:9]
        if use_cluster:
            expected_jobs = ExperimentExecutor._get_expected_jobs(jobs)
            pipeline = self._build_pipeline(expected_jobs)
            self._cluster.start()
            await pipeline.execute(jobs)
            self._cluster.stop()
        else:
            ExperimentExecutor._run_locally(jobs)

    def _build_pipeline(self, expected_jobs: ExpectedJobs) -> Pipeline:
        return Pipeline([
            Segment(
                name="Neural Trainer",
                capacity=50,
                expected_jobs=expected_jobs.training,
                submit=self._cluster.submit,
                pipe=lambda model_task: self._optimisation_pipe(model_task)
            ),
            Segment(
                name="Optimiser",
                capacity=200,
                expected_jobs=expected_jobs.optimisation,
                submit=self._cluster.submit
            )
        ])

    @staticmethod
    def _run_locally(jobs: List[Job]):
        container = JobContainer()
        container.init_resources()
        container.wire(modules=[__name__])
        for job in jobs:
            print(job.encode())
            job.run(container)

    @staticmethod
    def _get_expected_jobs(jobs: List[ModelJob]):
        training = len(jobs)
        optimisation = 0
        for job in jobs:
            optimisation += len(solvable_by(job.config.activation_fn))
        return ExpectedJobs(training, optimisation)

    async def _optimisation_pipe(self, model_task: Awaitable[str]) -> List[Job]:
        model_id = await model_task
        bounds = Bounds(0.2)
        model = self._neural_repo.get(model_id)
        activation = model.neural_config.activation_fn
        return [OptimisationJob(model_id, bounds, solver) for solver in solvable_by(activation)]

    def _get_initial_jobs(self, experiment: Experiment) -> List[ModelJob]:
        hyper_params = experiment.get_hyper_params()
        searcher = GridSearch()
        jobs = []
        for dataset_id in self._sample_repo.get_all_dataset_id():
            config_pool = searcher.get_sequence(hyper_params, experiment.type)
            for config in config_pool:
                job = ModelJob(dataset_id, config, experiment.exp_id)
                jobs.append(job)
        return jobs
