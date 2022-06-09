import os

import htcondor

class Cluster:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.log_dir = os.path.join(self.root_dir, "resources/logs")
        scheduler_daemon = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd)
        self.scheduler = htcondor.Schedd(scheduler_daemon)

    def get_job_config(self):
        return {
            "executable": "echo Hello",  # the program to run on the execute node
            "output": f"{self.log_dir}/stdout.log",
            "error": f"{self.log_dir}/stderr.log",
            "log": f"{self.log_dir}/metadata.log",
            "request_cpus": "1",
            "request_memory": "128MB",
            "request_disk": "128MB"
        }

    def exec(self, config):
        job = htcondor.Submit(config)
        result = self.scheduler.submit(job)
        print(result.cluster())



if __name__ == '__main__':
    cluster = Cluster("//")
    config = cluster.get_job_config()
    print(config)
