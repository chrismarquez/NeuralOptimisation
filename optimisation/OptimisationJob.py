import os

from sklearn import metrics

from batch.Job import Job
from models.FNN import FNN, Activation
from optimisation.Optimiser import Optimiser, Bounds


class OptimisationJob(Job):

    def __init__(self, id: str, function: str, nodes: int, depth: int, activation: Activation, input_bounds: Bounds):
        super(OptimisationJob, self).__init__()
        self.id = id
        self.function = function
        self.nodes = nodes
        self.depth = depth
        self.activation = activation
        self.net = FNN(nodes, depth, activation)
        self.input_bounds = input_bounds

    def run(self):
        optimiser = Optimiser.load(f"trained/{self.function}/{self.id}.pt", self.input_bounds, lambda: self.net)
        print(f"Size [{self.nodes}] Depth [{self.depth}]")
        x_opt, y_opt, z_opt = optimiser.solve()
        location_error = metrics.mean_squared_error([0.0, 0.0], [x_opt, y_opt], squared=False)
        optimum_error = metrics.mean_squared_error([0.0], [z_opt], squared=False)
        computation_time = optimiser.optimisation_time
        filename = f"trained/optimisation/{self.function}.csv"
        if not os.path.exists(filename):
            with open(filename, 'a') as f:
                f.write("id,x,y,location_error,optimum_error,computation_time\n")
        with open(filename, 'a') as f:
            model_params = f"{id},{x_opt},{y_opt},{location_error},{optimum_error},{computation_time}\n"
            f.write(model_params)