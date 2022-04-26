from __future__ import annotations

import os
import tempfile
from typing import Callable, Mapping, Tuple

import pandas as pd
import pyomo.core
import pyomo.environ as pyo
import torch.onnx
from omlt import OmltBlock  # Ignoring dependency resolution
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition

import functions
from models.FNN import FNN
# likely from an API design error, omlt.io requires the tensorflow module even if its not being used
from models.LoadableModule import LoadableModule

Bounds = Mapping[int, Tuple[float, float]]


class Optimiser:

    def __init__(self, network_definition: NetworkDefinition, solver_name: str = 'cbc'):
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
        self._solver = pyo.SolverFactory(solver_name)
        self.optimisation_time = 0.0

    def solve(self):
        results = self._solver.solve(self._model, tee=False, options={"threads": 12})
        self.optimisation_time = results['Solver'][0]['Wallclock time']
        return pyo.value(self._model.x), pyo.value(self._model.y), pyo.value(self._model.output)

    @staticmethod
    def load(path: str, input_bounds: Bounds, build_net: Callable[[], LoadableModule] = lambda: FNN.instantiate()) -> Optimiser:
        try:
            from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
            net = FNN.load(path, build_net)
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as file:
                Optimiser._onnx_export(net, file)
                write_onnx_model_with_bounds(file.name, None, input_bounds)
                network_definition = load_onnx_neural_network_with_bounds(file.name)
                return Optimiser(network_definition)
        except ModuleNotFoundError:
            print("TensorFlow is oddly needed for this module")
            pass

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

    [x_max] = [x_max for fn, x_max in functions.pool if fn == functions.sum_squares]

    input_bounds = {
        i: (-0.2, 0.2) for i in range(2)
    }

    print(input_bounds)

    optimiser = Optimiser.load("../trained/test/sum_squares.pt", input_bounds)

    values = optimiser.solve()

    print(values)

