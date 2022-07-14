from __future__ import annotations

import os
import tempfile
from typing import Callable

import pandas as pd
import pyomo.core
import pyomo.environ as pyo
import torch.onnx
from omlt import OmltBlock  # Ignoring dependency resolution
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition

from data import functions
from models.FNN import FNN
# likely from an API design error, omlt.io requires the tensorflow module even if its not being used
from models.LoadableModule import LoadableModule
from optimisation.Solver import Solver, LinearSolver
from repositories.db_models import NeuralModel, Bounds

from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds


class Optimiser:

    def __init__(self, network_definition: NetworkDefinition, solver_type: Solver = LinearSolver.CBC):
        model = pyo.ConcreteModel()

        model.net = OmltBlock()
        model.x = pyo.Var()
        model.y = pyo.Var()
        model.output = pyo.Var()

        formulation = FullSpaceNNFormulation(network_definition)
        model.net.build_formulation(formulation)

        for layer_id, layer in enumerate(network_definition.layers):
            print(f"{layer_id}\t{layer}\t{layer.activation}")

        @model.Constraint()
        def connect_output(m):
            return m.output == m.net.outputs[0]

        @model.Constraint()
        def connect_x(m):
            return m.x == m.net.inputs[0]

        @model.Constraint()
        def connect_y(m):
            return m.y == m.net.inputs[1]

        model.objective = pyo.Objective(expr=model.output, sense=pyomo.core.minimize)

        self._model = model
        self._solver = pyo.SolverFactory(solver_type.value)
        self.optimisation_time = 0.0

    def solve(self):
        results = self._solver.solve(self._model, tee=False, options={"threads": 12})
        self.optimisation_time = results['Solver'][0]['Wallclock time']
        return pyo.value(self._model.x), pyo.value(self._model.y), pyo.value(self._model.output)

    @staticmethod
    def load_from_path(path: str, input_bounds: Bounds, solver_type: Solver, build_net: Callable[[], LoadableModule]) -> Optimiser:
        net = build_net().load(path)
        return Optimiser._load(net, input_bounds, solver_type)

    @staticmethod
    def load_from_model(neural_model: NeuralModel, input_bounds: Bounds, solver_type: Solver) -> Optimiser:  # TODO: Support CNN
        _, _, net_size, depth, activation = neural_model.neural_config
        net = FNN(net_size, depth, activation).load_bytes(neural_model.model_data)
        return Optimiser._load(net, input_bounds, solver_type)

    @staticmethod
    def _load(net: LoadableModule, input_bounds: Bounds, solver_type: Solver) -> Optimiser:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as file:
            Optimiser._onnx_export(net, file)
            write_onnx_model_with_bounds(file.name, None, input_bounds.to_pyomo_bounds())
            network_definition = load_onnx_neural_network_with_bounds(file.name)
            return Optimiser(network_definition, solver_type)

    @staticmethod
    def finished_optimisations(function: str):
        filename = f"trained/optimisation/{function}.csv"
        if not os.path.exists(filename):
            return []
        df = pd.read_csv(filename, delimiter=',')
        return list(df['id'].values)

    @staticmethod
    def _onnx_export(net, file):
        dummy_input = net.dummy_input()
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        return torch.onnx.export(
            model=net,
            args=dummy_input,
            f=file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )


if __name__ == '__main__':
    from repositories.NeuralModelRepository import NeuralModelRepository

    [x_max] = [x_max for fn, x_max in functions.pool.items() if fn == functions.sum_squares]
    input_bounds: Bounds = Bounds(0.2)
    print(input_bounds)
    repo = NeuralModelRepository("mongodb://cloud-vm-42-88.doc.ic.ac.uk:27017/")
    model = repo.get("62b4969f79d0fbcab4b0ff0b")
    optimiser = Optimiser.load_from_model(model, input_bounds, solver_type=Solver.CBC)
    values = optimiser.solve()
    print(values)
