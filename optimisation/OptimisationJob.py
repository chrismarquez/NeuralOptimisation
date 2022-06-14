from dataclasses import dataclass

from sklearn import metrics

from cluster.Job import Job
from cluster.JobContainer import JobContainer
from optimisation.Optimiser import Optimiser, Bounds
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.db_models import OptimisationProperties


@dataclass
class OptimisationJob(Job):
    neural_repo: NeuralModelRepository
    model_id: str
    input_bounds: Bounds

    def run(self, container: JobContainer):
        neural_model = self.neural_repo.get(self.model_id)
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
        self.neural_repo.update(neural_model)
