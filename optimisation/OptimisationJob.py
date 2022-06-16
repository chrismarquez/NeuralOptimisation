from dataclasses import dataclass

from sklearn import metrics

from cluster.Job import Job, JobType
from cluster.JobContainer import JobContainer
from cluster.JobInit import init_job
from optimisation.Optimiser import Optimiser, Bounds
from repositories.db_models import OptimisationProperties


@dataclass
class OptimisationJob(Job):
    model_id: str
    input_bounds: Bounds

    def run(self, container: JobContainer):
        neural_repo = container.neural_repository()
        neural_model = neural_repo.get(self.model_id)
        optimiser = Optimiser.load_from_model(neural_model, self.input_bounds)
        _, _, nodes, depth, _ = neural_model.neural_config
        print(f"Size [{nodes}] Depth [{depth}]")
        x_opt, y_opt, z_opt = optimiser.solve()
        location_error = metrics.mean_squared_error([0.0, 0.0], [x_opt, y_opt], squared=False)
        optimum_error = metrics.mean_squared_error([0.0], [z_opt], squared=False)
        computation_time = optimiser.optimisation_time
        neural_model.optimisation_properties = OptimisationProperties(
            x_opt, y_opt, self.input_bounds, location_error, optimum_error, computation_time
        )
        neural_repo.update(neural_model)

    def as_command(self) -> str:
        return f"python3 -m optimisation.OptimisationJob --job {self.encode()}"

    def get_job_type(self) -> JobType:
        return "CPU"


if __name__ == '__main__':
    init_job("OptimisationJob")