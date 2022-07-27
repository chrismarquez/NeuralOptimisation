from typing import Tuple

from sklearn import metrics

from cluster.Job import Job, JobType, UnnecessaryJobException
from cluster.JobContainer import JobContainer
from cluster.JobInit import init_job
from optimisation.Optimiser import Optimiser, Bounds, OptimisationException
from optimisation.Solver import Solver
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.db_models import OptimisationProperties, NeuralModel


class OptimisationJob(Job):

    def __init__(self, model_id: str, input_bounds: Bounds, solver_type: Solver):
        super().__init__(model_id)
        self.input_bounds = input_bounds
        self.solver_type = solver_type

    def _run(self, container: JobContainer):
        neural_repo = container.neural_repository()
        neural_model = neural_repo.get(self.model_id)
        if self.optimisation_exists(neural_model):
            raise UnnecessaryJobException()
        optimiser = Optimiser.load_from_model(neural_model, self.input_bounds, self.solver_type)
        _, _, nodes, depth, _ = neural_model.neural_config  # TODO: Prepare for CNN
        print(f"Size [{nodes}] Depth [{depth}]")
        try:
            results = optimiser.solve()
            self._save_optimisation_results(neural_repo, neural_model, optimiser, results)
        except OptimisationException as err:
            opt_properties = self._failed_optimisation_properties(err.computation_time)
            neural_model.optimisation_properties.append(opt_properties)
            neural_repo.update(neural_model)

    def _failed_optimisation_properties(self, computation_time: float):
        return OptimisationProperties(
            float("inf"),
            float("inf"),
            self.input_bounds,
            self.solver_type,
            float("inf"),
            float("inf"),
            computation_time,
            successful=False
        )

    def _save_optimisation_results(
        self,
        neural_repo: NeuralModelRepository,
        neural_model: NeuralModel,
        optimiser: Optimiser,
        results: Tuple[float, float, float]
    ):
        x_opt, y_opt, z_opt = results
        location_error = metrics.mean_squared_error([0.0, 0.0], [x_opt, y_opt], squared=False)
        optimum_error = metrics.mean_squared_error([0.0], [z_opt], squared=False)
        computation_time = optimiser.optimisation_time
        opt_properties = OptimisationProperties(
            x_opt,
            y_opt,
            self.input_bounds,
            self.solver_type,
            location_error,
            optimum_error,
            computation_time,
            successful=True
        )
        neural_model.optimisation_properties.append(opt_properties)
        neural_repo.update(neural_model)

    def optimisation_exists(self, neural_model: NeuralModel) -> bool:
        existing_optimisations = [opt.solver_type for opt in neural_model.optimisation_properties]
        return self.solver_type in existing_optimisations

    def as_command(self) -> str:
        return f"python3 -m optimisation.OptimisationJob --job {self.encode()}"

    def get_job_type(self) -> JobType:
        return "CPU"

    def requires_gurobi_license(self) -> bool:
        return self.solver_type == "gurobi"


if __name__ == '__main__':
    init_job("OptimisationJob")
