import asyncio
import subprocess
from asyncio import Future
from typing import List, Awaitable

from cluster.Job import Job, JobType
from cluster.SlurmJobStatus import SlurmJobStatus, SlurmJobState
from cluster.WorkerPool import WorkerPool


class SlurmPool(WorkerPool):

    @staticmethod
    def _parse_job_id(result: str) -> int:
        raw_job_id = result.rstrip("\\n").split("Submitted batch job ")[-1]
        return int(raw_job_id)

    def __init__(self, root_dir: str, capacity: int):
        super().__init__(capacity, root_dir)

    async def submit(self, job: Job) -> Awaitable[str]:
        slurm_job_id = await self._submit(job)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = self._post_process(future, slurm_job_id)
        asyncio.create_task(task)
        return future

    async def _post_process(self, future: Future[str], slurm_job_id: int):
        while True:
            status = await self.status(slurm_job_id)
            print(f"Job Status {slurm_job_id}: {status}")
            if status.job_state == SlurmJobState.COMPLETED:
                break
            await asyncio.sleep(3)
        lines = self.get_job_output(slurm_job_id)
        model_id = SlurmPool.find_model_id(lines)
        future.set_result(model_id)

    def job_type(self) -> JobType:
        return "GPU"

    async def _submit(self, job: Job) -> int:
        await self._request_slot()
        script = self._to_runnable_script(job)
        sbatch = f"sbatch {script}"
        try:
            result = subprocess.run(sbatch, shell=True, capture_output=True)
            output = result.stdout.decode("utf-8")
            return SlurmPool._parse_job_id(output)
        except ValueError:
            await self._release_slot()
            raise RuntimeError(f"Task creation error for job {job.uuid}")

    async def status(self, job_id: int) -> SlurmJobStatus:
        cmd = f"scontrol show job <job_id>".replace("<job_id>", str(job_id))
        result = subprocess.run(cmd, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        status = SlurmJobStatus.from_log(output)
        if status.job_state == SlurmJobState.COMPLETED:
            await self._release_slot()
        return status

    def get_job_output(self, job_id: int) -> List[str]:
        file = f"{self.root_dir}/slurm20-{job_id}.out"
        with open(file) as f:
            return f.readlines()

    @staticmethod
    def find_model_id(lines: List[str]) -> str:
        return lines[-1].split("NEURAL_MODEL_ID:")[-1]


