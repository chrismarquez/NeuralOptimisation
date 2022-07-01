from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class CondorJobState(Enum):
    R = "RUNNING"
    I = "IDLE"
    H = "HOLD"


@dataclass
class CondorJobStatus:
    job_id: str
    user_id: str
    submit_time: str
    run_time: str
    job_state: CondorJobState
    cmd: str

    @staticmethod
    def from_log(log: str) -> List[CondorJobStatus]:
        lines = log.strip().split("\n")
        titles = lines[1].split()
        raw_records = lines[2:]
        records = []
        for record in raw_records:
            if record == "":
                break
            raw_fields = record.split()
            id_fields = raw_fields[:2]
            fields = id_fields + [" ".join(raw_fields[2:4])] + raw_fields[4:len(titles)]
            args = raw_fields[len(titles):]
            fields.append(" ".join(args))
            records.append(zip(titles, fields))

        all_attributes = [
            {
                key: val
                for (key, val) in rec
            } for rec in records
        ]

        return [
            CondorJobStatus(
                job_id=attributes["ID"],
                user_id=attributes["OWNER"],
                submit_time=attributes["SUBMITTED"],
                run_time=attributes["RUN_TIME"],
                job_state=CondorJobState[attributes["ST"]],
                cmd=attributes["CMD"]
            )
            for attributes in all_attributes
        ]


if __name__ == '__main__':
    with open("../resources/cluster/condor_q_running.txt") as f:
        raw = f.read()
        print(CondorJobStatus.from_log(raw))
