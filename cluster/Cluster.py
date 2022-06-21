from time import sleep

import asyncio
from typing import Awaitable, List

from cluster.Job import Job
from cluster.SlurmPool import SlurmPool


class Cluster:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.slurm_pool = SlurmPool(root_dir, capacity=2)
        pass

    # Coroutine launch method
    async def submit(self, job: Job) -> Awaitable[str]:
        job_type = job.get_job_type()
        if job_type == "CPU":
            pass
        elif job_type == "GPU":
            task = self._submit_slurm(job)
            return asyncio.create_task(task)

    async def _submit_slurm(self, job: Job) -> str:
        slurm_job_id = await self.slurm_pool.submit(job)
        while True:
            status = await self.slurm_pool.status(slurm_job_id)
            if status.job_state == "COMPLETE":
                break
            await asyncio.sleep(3)
        lines = self.slurm_pool.get_job_output(slurm_job_id)
        return Cluster.find_model_id(lines)

    @staticmethod
    def find_model_id(lines: List[str]) -> str:
        return lines[-1].split("NEURAL_MODEL_ID:")[-1]


if __name__ == '__main__':

    async def main():
        import os
        from experiments.ExperimentExecutor import ExperimentExecutor
        from repositories.SampleDatasetRepository import SampleDatasetRepository
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/cluster")[0]
        cluster = Cluster(ROOT_DIR)
        sample = SampleDatasetRepository("mongodb://cloud-vm-42-88.doc.ic.ac.uk:27017/")
        executor = ExperimentExecutor(cluster, sample)
        job = executor._get_initial_jobs()[0]
        job_id = await cluster.submit(job)
        print(job_id)
        sleep(2)
        # status = cluster.slurm_pool.status(job_id)
        # print(status)

    asyncio.run(main())
