import os
from typing import List

import pandas as pd
from tqdm import tqdm

from batch.Executor import Executor
from batch.Job import Job
from optimisation.OptimisationJob import OptimisationJob
from optimisation.Optimiser import Optimiser


class OptimisationExecutor(Executor):

    def __init__(self):
        super().__init__()

    def _get_jobs(self) -> List[Job]:
        path = "trained/metadata"
        input_bounds = {
            i: (-0.2, 0.2) for i in range(2)
        }
        jobs = []
        for file in os.listdir(path):
            function, _ = file.split(".")
            finished = Optimiser.finished_optimisations(function)
            df = pd.read_csv(f"{path}/{file}", delimiter=",")
            df.sort_values(by=["network_size", "depth"])
            df = df[df["activation_fn"] == "ReLU"]
            for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                id, _, _, nodes, depth, activation_fn, _, _ = row
                if id not in finished and depth == 2:  # Check depth vanishing
                    job = OptimisationJob(id, function, nodes, depth, activation_fn, input_bounds)
                    jobs.append(job)
                else:
                    print(f"Skip {id}")
        return jobs

