from __future__ import annotations

from typing import List, Union, Literal

from models.LoadableModule import Activation


def solvable_by(activation: Activation) -> List[Solver]:
    if activation == "ReLU":
        return ["cbc", "gurobi"]
    else:
        return ["ipopt"] # TODO: Add another NLP Solver


LinearSolver = Literal["cbc", "gurobi"]
NonLinearSolver = Literal["ipopt"]

Solver = Union[LinearSolver, NonLinearSolver]
