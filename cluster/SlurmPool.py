import asyncio
import subprocess
from asyncio import Future, Task
from typing import List, Awaitable, Dict

from cluster.Job import Job, JobType
from cluster.SlurmJobStatus import SlurmJobStatus, SlurmJobState
from cluster.WorkerPool import WorkerPool


class SlurmPool(WorkerPool):

    @staticmethod
    def _parse_job_id(result: str) -> int:
        raw_job_id = result.rstrip("\\n").split("Submitted batch job ")[-1]
        return int(raw_job_id)

    def __init__(self, root_dir: str, capacity: int, debug: bool):
        super().__init__(capacity, root_dir)
        self.tasks: Dict[Future, Task] = {}
        self.debug = debug

    async def submit(self, job: Job) -> Awaitable[bool]:
        slurm_job_id = await self._submit(job)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = self._post_process(future, slurm_job_id)
        self.tasks[future] = asyncio.create_task(task)
        return future

    async def _post_process(self, future: Future[bool], slurm_job_id: int):
        while True:
            try:
                status = await self.status(slurm_job_id)
                if self.debug:
                    print(f"Job Status {slurm_job_id}: {status}")
                if status.job_state == SlurmJobState.COMPLETED:
                    await self._release_slot()
                    self.tasks.pop(future)
                    break
            except KeyError:
                pass
            await asyncio.sleep(3)
        future.set_result(True)

    def job_type(self) -> JobType:
        return "GPU"

    async def _submit(self, job: Job) -> int:
        await self._request_slot()
        script = self._runnable_script_from(job)
        sbatch = f"sbatch {script}"
        try:
            result = subprocess.run(sbatch, shell=True, capture_output=True)
            output = result.stdout.decode("utf-8")
            return SlurmPool._parse_job_id(output)
        except ValueError as err:
            await self._release_slot()
            if self.debug:
                print(err)
            raise RuntimeError(f"Task creation error for job {job.uuid}")

    async def status(self, job_id: int) -> SlurmJobStatus:
        cmd = f"scontrol show job <job_id>".replace("<job_id>", str(job_id))
        result = subprocess.run(cmd, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        return SlurmJobStatus.from_log(output)

    def get_job_output(self, job_id: int) -> List[str]:
        file = f"{self.root_dir}/slurm20-{job_id}.out"
        with open(file) as f:
            return f.readlines()


