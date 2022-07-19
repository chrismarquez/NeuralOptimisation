from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SlurmJobState(Enum):
    COMPLETED = "COMPLETED"
    COMPLETING = "COMPLETING"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"


@dataclass
class SlurmJobStatus:
    job_id: str
    user_id: str
    job_state: SlurmJobState
    submit_time: str
    run_time: str
    std_out: str
    std_err: str
    cmd: str
    work_dir: str
    node: Optional[str]
    exit_code: Optional[str]

    @staticmethod
    def from_log(log: str) -> SlurmJobStatus:
        raw_pairs = [item for item in log.replace("\n", "").split(" ") if (item != "" and "=" in item)]
        pairs = [(i.split("=")[0], i.split("=")[1]) for i in raw_pairs]
        attributes = {
            key: val
            for (key, val) in pairs
        }
        return SlurmJobStatus(
            job_id=attributes["JobId"],
            user_id=attributes["UserId"],
            job_state=SlurmJobState[attributes['JobState']],
            submit_time=attributes["SubmitTime"],
            run_time=attributes["RunTime"],
            std_out=attributes["StdOut"],
            std_err=attributes["StdErr"],
            cmd=attributes["Command"],
            work_dir=attributes["WorkDir"],
            node=attributes.get("BatchHost", None),
            exit_code=attributes.get("ExitCode", None)
        )


if __name__ == '__main__':
    with open("../resources/cluster/scontrol_running.txt") as f:
        raw = f.read()
        print(SlurmJobStatus.from_log(raw))
