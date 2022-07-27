from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Callable

import pandas as pd
import pyomo.core
import pyomo.environ as pyo
import torch.onnx
from omlt import OmltBlock  # Ignoring dependency resolution
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition

from data import functions
from models.CNN import CNN
from models.FNN import FNN
# likely from an API design error, omlt.io requires the tensorflow module even if its not being used
from models.LoadableModule import LoadableModule
from optimisation.Solver import Solver
from repositories.db_models import NeuralModel, Bounds, FeedforwardNeuralConfig, ConvolutionalNeuralConfig

_MINUTE = 60.0
_HOUR = 60.0 * _MINUTE


class OptimisationException(Exception):

    def __init__(self, computation_time: float):
        self.computation_time = computation_time


class Optimiser:

    def __init__(self, network_definition: NetworkDefinition, solver_type: Solver = "cbc"):
        self.solver_type = solver_type
        self.timeout = 3 * _HOUR

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
        self._solver = pyo.SolverFactory(solver_type)
        self.optimisation_time = 0.0

    def solve(self):
        options = {} if self.solver_type == "ipopt" else {"threads": 8}
        start_time = datetime.now()
        try:
            results = self._solver.solve(self._model, tee=True, timelimit=self.timeout, options=options)
            self.optimisation_time = self._get_optimisation_time(results)
            return pyo.value(self._model.x), pyo.value(self._model.y), pyo.value(self._model.output)
        except Exception:
            end_time = datetime.now()
            duration = end_time - start_time
            raise OptimisationException(duration.total_seconds())

    @staticmethod
    def load_from_path(path: str, input_bounds: Bounds, solver_type: Solver,
                       build_net: Callable[[], LoadableModule]) -> Optimiser:
        net = build_net().load(path)
        return Optimiser._load(net, input_bounds, solver_type)

    @staticmethod
    def load_from_model(neural_model: NeuralModel, input_bounds: Bounds, solver_type: Solver) -> Optimiser:
        if type(neural_model.neural_config) is FeedforwardNeuralConfig:
            _, _, net_size, depth, activation = neural_model.neural_config
            net = FNN(net_size, depth, activation).load_bytes(neural_model.model_data)
        elif type(neural_model.neural_config) is ConvolutionalNeuralConfig:
            net = CNN(
                start_size=neural_model.neural_config.start_size,
                filter_size=neural_model.neural_config.filter_size,
                filters=neural_model.neural_config.filters,
                depth=neural_model.neural_config.depth,
                activation=neural_model.neural_config.activation_fn
            ).load_bytes(neural_model.model_data)
        else:
            raise RuntimeError("Unrecognized Network Type")

        return Optimiser._load(net, input_bounds, solver_type)

    def _get_optimisation_time(self, results) -> float:
        if self.solver_type == "cbc":
            return float(results['Solver'][0]['Wallclock time'])
        elif self.solver_type == "ipopt":
            return float(results['Solver'][0]['Time'])
        elif self.solver_type == "gurobi":
            return float(results['Solver'][0]['Wall time'])
        return -1.0

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
    model = repo.get("62deef0727dc2b9bd660c62b")
    optimiser = Optimiser.load_from_model(model, input_bounds, solver_type="gurobi")
    values = optimiser.solve()
    print(values)
