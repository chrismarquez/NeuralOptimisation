import subprocess
from time import sleep

import JobStatus


class Cluster:

    def __init__(self):
        pass

    @staticmethod
    def _parse_job_id(result: str) -> int:
        raw_job_id = result.rstrip("\\n").split("Submitted batch job ")[-1]
        return int(raw_job_id)

    def submit(self) -> int:
        cmd = f"sbatch slurmseg.sh"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        print(output)
        return Cluster._parse_job_id(output)

    def status(self, job_id: int) -> JobStatus:
        cmd = f"scontrol show job <job_id>".replace("<job_id>", str(job_id))
        result = subprocess.run(cmd, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        return JobStatus.from_log(output)


if __name__ == '__main__':
    cluster = Cluster()
    job_id = cluster.submit()
    print(job_id)
    sleep(1)
    status = cluster.status(job_id)
    print(status)