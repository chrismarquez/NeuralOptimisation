import asyncio
import inspect
import subprocess
import tempfile
from typing import List

from cluster.Job import Job
from cluster.JobStatus import JobStatus
from cluster.WorkerPool import WorkerPool


class SlurmPool(WorkerPool):

    def __init__(self, root_dir, capacity: int):
        super().__init__(capacity)
        self.root_dir = root_dir

    @staticmethod
    def _parse_job_id(result: str) -> int:
        raw_job_id = result.rstrip("\\n").split("Submitted batch job ")[-1]
        return int(raw_job_id)

    async def submit(self, job: Job) -> int:
        await self._request_slot()
        cmd = inspect.cleandoc(
            f"""
                #!/bin/bash
                source {self.root_dir}/venv/bin/activate
                APP_ENV=PROD {job.as_command()}
            """
        )
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="w") as file:
            file.write(cmd)
            script = file.name
            print(script)
        sbatch = f"sbatch {script}"
        await asyncio.sleep(15)
        try:
            result = subprocess.run(sbatch, shell=True, capture_output=True)
            output = result.stdout.decode("utf-8")
            return SlurmPool._parse_job_id(output)
        except ValueError:
            await self._release_slot()
            raise RuntimeError(f"Task creation error for job {job.uuid}")

    async def status(self, job_id: int) -> JobStatus:
        cmd = f"scontrol show job <job_id>".replace("<job_id>", str(job_id))
        result = subprocess.run(cmd, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        status = JobStatus.from_log(output)
        if status.job_state == "COMPLETE":
            await self._release_slot()
        return status

    def get_job_output(self, job_id: int) -> List[str]:
        file = f"{self.root_dir}/slurm20-{job_id}.out"
        with open(file) as f:
            return f.readlines()


