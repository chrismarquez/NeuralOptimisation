import inspect
import subprocess
import tempfile
from time import sleep


from JobStatus import JobStatus


class Cluster:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        pass

    @staticmethod
    def _parse_job_id(result: str) -> int:
        raw_job_id = result.rstrip("\\n").split("Submitted batch job ")[-1]
        return int(raw_job_id)

    def submit(self, target: str) -> int:
        cmd = inspect.cleandoc(f"""
            #!/bin/bash
            source {self.root_dir}/venv/bin/activate
            python3 {self.root_dir}/{target}
        """)
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="w") as file:
            file.write(cmd)
            script = file.name
            print(script)
        sbatch = f"sbatch {script}"
        result = subprocess.run(sbatch, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        return Cluster._parse_job_id(output)

    def status(self, job_id: int) -> JobStatus:
        cmd = f"scontrol show job <job_id>".replace("<job_id>", str(job_id))
        result = subprocess.run(cmd, shell=True, capture_output=True)
        output = result.stdout.decode("utf-8")
        return JobStatus.from_log(output)


if __name__ == '__main__':
    import os
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/cluster")[0]
    cluster = Cluster(ROOT_DIR)
    job_id = cluster.submit("models/ModelJob.py")
    print(job_id)
    sleep(1)
    status = cluster.status(job_id)
    print(status)
