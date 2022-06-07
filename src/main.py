
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from models.ModelsExecutor import ModelsExecutor
from optimisation.OptimisationExecutor import OptimisationExecutor
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository


class Container(containers.DeclarativeContainer):
    path = "resources/config.ini"
    config = providers.Configuration(ini_files=[path])

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    # Executors

    optimisation_executor = providers.Singleton(OptimisationExecutor, repository=neural_repository)
    models_executor = providers.Singleton(ModelsExecutor, neural_repo=neural_repository, sample_repo=sample_repository)


@inject
def main(container: Container = Provide[Container]):
    optimisation_executor = container.optimisation_executor()
    optimisation_executor.run_all_jobs()
    print("Hello there")

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from src.cluster.Cluster import Cluster
from src.models.ModelsExecutor import ModelsExecutor
from src.optimisation.OptimisationExecutor import OptimisationExecutor
from src.repositories.NeuralModelRepository import NeuralModelRepository
from src.repositories.SampleDatasetRepository import SampleDatasetRepository

from constants import ROOT_DIR


class Container(containers.DeclarativeContainer):
    path = "resources/config.ini"
    config = providers.Configuration(ini_files=[path])

    # Cluster

    cluster = providers.Singleton(Cluster, root_dir=ROOT_DIR)

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    # Executors

    optimisation_executor = providers.Singleton(OptimisationExecutor, repository=neural_repository)
    models_executor = providers.Singleton(ModelsExecutor, neural_repo=neural_repository, sample_repo=sample_repository)


@inject
def main(container: Container = Provide[Container]):
    #optimisation_executor = container.optimisation_executor()
    #optimisation_executor.run_all_jobs()
    #print("Hello there")
    cluster = container.cluster()
    config = cluster.get_job_config()
    print(config)


if __name__ == '__main__':
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    main()
