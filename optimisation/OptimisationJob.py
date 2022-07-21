from dataclasses import dataclass

from sklearn import metrics

from cluster.Job import Job, JobType
from cluster.JobContainer import JobContainer
from cluster.JobInit import init_job
from optimisation.Optimiser import Optimiser, Bounds
from optimisation.Solver import Solver
from repositories.db_models import OptimisationProperties, NeuralModel


@dataclass
class OptimisationJob(Job):
    model_id: str
    input_bounds: Bounds
    solver_type: Solver

    def run(self, container: JobContainer):
        neural_repo = container.neural_repository()
        neural_model = neural_repo.get(self.model_id)
        if self.optimisation_exists(neural_model):
            return
        optimiser = Optimiser.load_from_model(neural_model, self.input_bounds, self.solver_type)
        _, _, nodes, depth, _ = neural_model.neural_config
        print(f"Size [{nodes}] Depth [{depth}]")
        x_opt, y_opt, z_opt = optimiser.solve()
        location_error = metrics.mean_squared_error([0.0, 0.0], [x_opt, y_opt], squared=False)
        optimum_error = metrics.mean_squared_error([0.0], [z_opt], squared=False)
        computation_time = optimiser.optimisation_time
        neural_model.optimisation_properties.append(
            OptimisationProperties(
                x_opt, y_opt, self.input_bounds, self.solver_type, location_error, optimum_error, computation_time
            )
        )
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
