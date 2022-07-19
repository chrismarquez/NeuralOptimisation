import asyncio

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from cluster.CondorPool import CondorConfig, CondorPool
from cluster.JobContainer import JobContainer
from experiments.Experiment import Experiment
from experiments.ExperimentExecutor import ExperimentExecutor
from cluster.Cluster import Cluster
from optimisation.OptimisationJob import OptimisationJob
from optimisation.Solver import LinearSolver, NonLinearSolver
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository

from constants import ROOT_DIR, get_env, get_config


# TODO: Put relevant Containers to dependency inject both jobs inside cluster and the main driver program
from repositories.db_models import Bounds


class Container(containers.DeclarativeContainer):
    print("Initialising dependencies...")

    env = get_env()
    config_file = get_config(env)
    path = f"resources/{config_file}"
    config = providers.Configuration(ini_files=[path])

    # Cluster

    cluster = providers.Singleton(Cluster, root_dir=ROOT_DIR)

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    # Executors

    experiment_executor = providers.Singleton(
        ExperimentExecutor,
        cluster=cluster,
        neural_repo=neural_repository,
        sample_repo=sample_repository
    )

    print("Dependencies ready.")


@inject
async def main(container: Container = Provide[Container]):
    executor = container.experiment_executor()
    experiment = Experiment("test-2", "Feedforward")
    await executor.run_experiment(experiment, test_run=True)


if __name__ == '__main__':
    print("Loading Container...")
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    print("Container ready.")
    asyncio.run(main())
