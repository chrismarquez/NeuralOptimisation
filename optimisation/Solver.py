from __future__ import annotations

from enum import Enum
from typing import List, cast

from models.LoadableModule import Activation


def solvable_by(activation: Activation) -> List[Solver]:
    if activation == "ReLU":
        solver_type = LinearSolver
    else:
        solver_type = NonLinearSolver
    return [cast(Solver, solver) for solver in solver_type]


class Solver(Enum):
    pass


class LinearSolver(Solver):
    CBC = "cbc"
    GUROBI = "gurobi"


class NonLinearSolver(Solver):
    IPOPT = "ipopt"
    GUROBI = "gurobi"

