from dataclasses import dataclass
from typing import List, Awaitable

from tqdm import tqdm

from cluster.Cluster import Cluster
from cluster.Job import Job
from cluster.JobInit import init_container
from cluster.Pipeline import Segment, Pipeline
from experiments.Experiment import Experiment
from models.GridSearch import GridSearch
from models.ModelJob import ModelJob
from optimisation.OptimisationJob import OptimisationJob
from optimisation.Solver import solvable_by
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository
from repositories.db_models import Bounds, NeuralModel


@dataclass
class ExpectedJobs:
    training: int
    optimisation: int


class ExperimentExecutor:

    def __init__(
        self,
        cluster: Cluster,
        neural_repo: NeuralModelRepository,
        sample_repo: SampleDatasetRepository,
        raw_debug: str
    ):
        self._cluster = cluster
        self._neural_repo = neural_repo
        self._sample_repo = sample_repo
        self.debug = raw_debug == "True"

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
        return Pipeline(debug=self.debug, segments=[
            Segment(
                name="Neural Trainer",
                capacity=50,
                expected_jobs=expected_jobs.training,
                submit=self._cluster.submit,
                pipe=lambda completed_job, model_task: self._optimisation_pipe(completed_job, model_task)
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
        print("Initialising Job Container")
        container = init_container()
        for job in jobs:
            job.run(container)

    @staticmethod
    def _get_expected_jobs(jobs: List[ModelJob]):
        training = len(jobs)
        optimisation = 0
        for job in jobs:
            optimisation += len(solvable_by(job.config.activation_fn))
        return ExpectedJobs(training, optimisation)

    async def _optimisation_pipe(self, completed_job: Job, model_task: Awaitable[bool]) -> List[Job]:
        await model_task
        bounds = Bounds(0.2)
        model_id = completed_job.model_id
        model = self._neural_repo.get(model_id)
        activation = model.neural_config.activation_fn
        return [OptimisationJob(model_id, bounds, solver) for solver in solvable_by(activation)]

    def _get_initial_jobs(self, experiment: Experiment) -> List[ModelJob]:
        experiment_exists = self._neural_repo.experiment_exists(experiment.exp_id)
        if not experiment_exists:
            return self._init_experiment(experiment)
        neural_models = self._neural_repo.get_all(experiment.exp_id)
        neural_models = self._calculate_execution(neural_models, experiment)
        return [ModelJob(model.id, model.neural_config, experiment.epochs) for model in neural_models]

    def _calculate_execution(self, neural_models: List[NeuralModel], experiment: Experiment) -> List[NeuralModel]:
        total_models = len(neural_models)
        to_train = self._neural_repo.count_models_to_train(experiment.exp_id)
        trained_models = total_models - to_train
        print(f"{trained_models} / {total_models} already trained. {to_train} models remain to be trained.")
        incomplete_models = [model for model in neural_models if not model.is_complete()]
        print(f"{len(incomplete_models)} / {total_models} are incomplete. {incomplete_models} models will be added to pipeline.")
        return incomplete_models

    def _init_experiment(self, exp: Experiment) -> List[ModelJob]:
        hyper_params = exp.get_hyper_params()
        searcher = GridSearch()
        jobs = []
        config_pool = searcher.get_sequence(hyper_params, exp.type)
        for dataset_id in self._sample_repo.get_all_dataset_id():
            function_name = self._sample_repo.get_name_by_id(dataset_id)
            model_list = [
                NeuralModel(
                    function=function_name,
                    type=config.get_neural_type(),
                    neural_config=config,
                    expected_optimisations=len(solvable_by(config.activation_fn)),
                    experiment_id=exp.exp_id
                ) for config in config_pool
            ]
            with tqdm(model_list) as models:
                models.set_description(f"Creating Experiment {exp.exp_id} models for function {function_name}")
                for model in models:
                    model_id = self._neural_repo.save(model)
                    job = ModelJob(model_id, model.neural_config, exp.epochs)
                    jobs.append(job)
        return jobs
