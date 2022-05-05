
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from src.models.ModelsExecutor import ModelsExecutor
from src.optimisation.OptimisationExecutor import OptimisationExecutor
from src.repositories.NeuralModelRepository import NeuralModelRepository
from src.repositories.SampleDatasetRepository import SampleDatasetRepository


class Container(containers.DeclarativeContainer):
    # config = providers.Configuration(ini_files=["config.ini"])

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository)
    sample_repository = providers.Singleton(SampleDatasetRepository)

    # Executors

    optimisation_executor = providers.Singleton(OptimisationExecutor, repository=neural_repository)
    models_executor = providers.Singleton(ModelsExecutor, neural_repo=neural_repository, sample_repo=sample_repository)


@inject
def main(container: Container = Provide[Container]):
    optimisation_executor = container.models_executor()
    optimisation_executor.run_all_jobs()
    print("Hello there")


if __name__ == '__main__':
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    main()
