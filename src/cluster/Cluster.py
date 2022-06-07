import os

import htcondor
import sys

class Cluster:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.log_dir = os.path.join(self.root_dir, "resources/logs")

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


if __name__ == '__main__':
    cluster = Cluster("/home/christopher/PycharmProjects/neuralOptimisation/")
    config = cluster.get_job_config()
    print(config)
