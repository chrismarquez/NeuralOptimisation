import os

import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from models.FNN import FNN
from optimisation.Optimiser import Optimiser


class Executor:

    def __init__(self):
        pass


    def optimise_all_models(self):
        path = "trained/metadata"
        input_bounds = {
            i: (-0.2, 0.2) for i in range(2)
        }

        for file in os.listdir(path):
            function, _ = file.split(".")
            finished = Optimiser.finished_optimisations(function)
            df = pd.read_csv(f"{path}/{file}", delimiter=",")
            df.sort_values(by=["network_size", "depth"])
            df = df[df["activation_fn"] == "ReLU"]
            for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                id, _, _, nodes, depth, activation_fn, _, _ = row
                if id not in finished and depth == 2:
                    net = FNN(nodes, depth, activation_fn)
                    optimiser = Optimiser.load(f"trained/{function}/{id}.pt", input_bounds, lambda: net)
                    print(f"Size [{nodes}] Depth [{depth}]")
                    x_opt, y_opt, z_opt = optimiser.solve()
                    location_error = metrics.mean_squared_error([0.0, 0.0], [x_opt, y_opt], squared=False)
                    optimum_error = metrics.mean_squared_error([0.0], [z_opt], squared=False)
                    computation_time = optimiser.optimisation_time
                    filename = f"trained/optimisation/{function}.csv"
                    if not os.path.exists(filename):
                        with open(filename, 'a') as f:
                            f.write("id,x,y,location_error,optimum_error,computation_time\n")
                    with open(filename, 'a') as f:
                        model_params = f"{id},{x_opt},{y_opt},{location_error},{optimum_error},{computation_time}\n"
                        f.write(model_params)
                else:
                    print(f"Skip {id}")