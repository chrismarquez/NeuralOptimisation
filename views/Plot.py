import os
from dataclasses import dataclass, astuple
from typing import Callable, List, Optional, Literal, Union, Dict

import numpy as np
import pandas as pd
from dacite import from_dict
from matplotlib import pyplot as plt

from constants import ROOT_DIR
from data import functions
from data.functions import Function2D
from models.FNN import FNN
from models.GridSearch import GridSearch
from models.LoadableModule import Activation
from optimisation.Solver import solvable_by
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.db_models import NeuralModel, OptimisationProperties

Transform = Callable[[float], float]
Filter = Callable[[NeuralModel], bool]

@dataclass
class PlotSettings:
    function: str
    depth: int
    activation: Activation

    def generate_signature(self) -> str:
        return f"{self.function}-D{self.depth}-A{self.activation}"


@dataclass
class PlotLabels:
    titles: Union[str, List[str]]
    x_axis: str
    y_axis: str
    plot_type: str


@dataclass
class Plot:
    exp_id: str
    settings: PlotSettings
    data: pd.DataFrame

    @staticmethod
    def load_data(
        models: List[NeuralModel],
        function: Optional[str] = None,
        filter: Optional[Filter] = None
    ) -> pd.DataFrame:
        models = [model for model in models if model.is_complete()]
        if function is not None:
            models = [model for model in models if model.function == function]
        if filter is not None:
            models = [model for model in models if filter(model)]
        view_models = [
            {
                "id": model.id,
                "network_size": model.neural_config.network_size,
                "depth": model.neural_config.depth,
                "function": model.function,
                "activation_fn": model.neural_config.activation_fn,
                "learning_rate": model.neural_config.learning_rate,
                "batch_size": model.neural_config.batch_size,
                "r2": model.neural_properties.r2,
                "rmse": model.neural_properties.rmse,
                "optimisation_time_base": Plot._get_optimisation(model, "base").computation_time,
                "optimisation_error_base": Plot._get_optimisation(model, "base").location_error,
                "optimisation_time_alt": Plot._get_optimisation(model, "alt").computation_time,
                "optimisation_error_alt": Plot._get_optimisation(model, "alt").location_error,
                "optimisation_obj_error_base": Plot._get_optimisation(model, "base").optimum_error,
                "optimisation_obj_error_alt": Plot._get_optimisation(model, "alt").optimum_error,
            }
            for model in models
        ]
        df = pd.DataFrame(view_models)
        return df

    @staticmethod
    def _get_optimisation(model: NeuralModel, name: Literal["base", "alt"]) -> OptimisationProperties:
        [base, alt] = solvable_by(model.neural_config.activation_fn)
        if name == "base":
            return [m for m in model.optimisation_properties if m.solver_type == base][0]
        else:
            return [m for m in model.optimisation_properties if m.solver_type == alt][0]

    @staticmethod
    def load_data_from_file(function: str) -> pd.DataFrame:
        df_training = pd.read_csv(f"{ROOT_DIR}/resources/trained/metadata/{function}.csv", delimiter=",")
        df_optimisation = pd.read_csv(f"{ROOT_DIR}/resources/trained/optimisation/{function}.csv", delimiter=",")
        df = df_training.set_index('id').join(df_optimisation.set_index('id'))
        df = df[df["activation_fn"] == "ReLU"]
        return df

    @staticmethod
    def box_plot(
        x: List,
        y: List,
        labels: PlotLabels,
        exp_id: str,
        x_transform: Transform = lambda x: x,
        save_name: Optional[str] = None
    ):
        buckets = {}
        for (x_coord, y_coord) in zip(x, y):
            params_class = x_transform(x_coord)
            if params_class not in buckets:
                buckets[params_class] = []
            if not np.isnan(y_coord):
                buckets[params_class].append(y_coord)
        axes, values = zip(*list(sorted(buckets.items())))
        figure, ax = plt.subplots(figsize=(10, 6))
        ax.set_axisbelow(True)
        ax.set_title(labels.titles)
        ax.set_xlabel(labels.x_axis)
        ax.set_ylabel(labels.y_axis)
        xtickNames = plt.setp(ax, xticklabels=axes)
        plt.setp(xtickNames, rotation=45, fontsize=8)
        ax.boxplot(values)
        #figure.show()
        if save_name is not None:
            Plot._save_plot(labels, save_name, exp_id, figure)

    @staticmethod
    def dual_scatter_plot(
        x: List,
        y1: List,
        y2: List,
        labels: PlotLabels,
        exp_id: str,
        save_name: Optional[str] = None
    ):
        figure, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
        title1, title2 = labels.titles
        Plot._prepare_axis(ax1, x, y1, title1, labels)
        Plot._prepare_axis(ax2, x, y2, title2, labels)
        plt.tight_layout()
        #figure.show()
        if save_name is not None:
            Plot._save_plot(labels, save_name, exp_id, figure)

    @staticmethod
    def weights_histogram(model: NeuralModel, exp_id: str, save_name: Optional[str] = None):
        _, _, net_size, depth, activation = model.neural_config
        net = FNN(net_size, depth, activation).load_bytes(model.model_data)
        history = []
        params = list(net.parameters())
        for param in params:
            history += list(param.detach().flatten().numpy())
        fig, ax = plt.subplots(figsize=(8, 6))

        # ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, 2_500])

        # the histogram of the data
        num_bins = 100
        _ = ax.hist(history, num_bins)
        fig.tight_layout()
        #fig.show()
        labels = PlotLabels(
            titles=f"Rastrigin Model Histogram",
            x_axis="Buckets for parameter values",
            y_axis="Neurons count per bucket",
            plot_type="histogram"
        )
        if save_name is not None:
            Plot._save_plot(labels, save_name, exp_id, fig)

    @staticmethod
    def _save_plot(labels: PlotLabels, save_name: str, exp_id: str, figure: plt.Figure):
        exp_dir = f"{ROOT_DIR}/resources/plots/{exp_id}"
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        plot_dir = f"{exp_dir}/{labels.plot_type}"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        l.append(f"./figures/plots/{exp_id}/{labels.plot_type}/{save_name}.svg\n")
        figure.savefig(f"{plot_dir}/{save_name}.svg", dpi=600)

    @staticmethod
    def _prepare_axis(axes: plt.Axes, x: List, y: List, title: str, labels: PlotLabels):
        # axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_axisbelow(True)
        axes.set_title(title)
        axes.set_xlabel(labels.x_axis)
        axes.set_ylabel(labels.y_axis)
        axes.scatter(x, y)
        z = np.polyfit(x, np.log10(y), 1)
        p = np.poly1d(z)
        x_t = np.array(sorted(set(x)))
        axes.plot(x_t, 10 ** p(x_t), color='red')

    def __iter__(self):
        return iter(astuple(self))

    def plot_size_optimisation_time(self):
        function = self.settings.function
        size, depth = self.data['network_size'].values, self.data['depth'].values
        activation = self.data["activation_fn"].values
        x = [FNN(nodes=s, depth=d, activation=a).count_parameters() for (s, d, a) in zip(size, depth, activation)]
        y = self.data['optimisation_time_base'].values
        labels = PlotLabels(
            titles=f"Optimisation Time / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="Time (in seconds)",
            plot_type="opt_time-slearnable_params"
        )
        save_name = self.settings.generate_signature()
        Plot.box_plot(x, y, labels, self.exp_id, lambda x_coord: round(x_coord / 10_000.0) * 10_000, save_name)

    def plot_size_opt_error(self):
        function = self.settings.function
        size, depth = self.data['network_size'].values, self.data['depth'].values
        activation = self.data["activation_fn"].values
        x = [FNN(nodes=s, depth=d, activation=a).count_parameters() for (s, d, a) in zip(size, depth, activation)]
        y = self.data['optimisation_error_base'].values
        labels = PlotLabels(
            titles=f"Optimisation Error / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of Optimisation result",
            plot_type="opt_error-learnable_params"
        )
        save_name = self.settings.generate_signature()
        Plot.box_plot(x, y, labels, self.exp_id, lambda x_coord: round(x_coord / 10_000.0) * 10_000, save_name)

        # figure = plt.figure()
        # figure.suptitle(f"Optimisation Error / Trainable Parameters [{function.capitalize()}]", fontsize=16)
        # plt.scatter(x, y)
        # figure.show()

    def plot_size_opt_obj_error(self):
        function = self.settings.function
        size, depth = self.data['network_size'].values, self.data['depth'].values
        activation = self.data["activation_fn"].values
        x = [FNN(nodes=s, depth=d, activation=a).count_parameters() for (s, d, a) in zip(size, depth, activation)]
        y = self.data['optimisation_obj_error_base'].values
        labels = PlotLabels(
            titles=f"Optimisation Objective Error / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of Optimal Objective Function Value",
            plot_type="opt_obj_error-learnable_params"
        )
        save_name = self.settings.generate_signature()
        Plot.box_plot(x, y, labels, self.exp_id, lambda x_coord: round(x_coord / 10_000.0) * 10_000, save_name)

    def plot_size_rmse(self):
        function = self.settings.function
        size, depth = self.data['network_size'].values, self.data['depth'].values
        activation = self.data["activation_fn"].values
        x = [FNN(nodes=s, depth=d, activation=a).count_parameters() for (s, d, a) in zip(size, depth, activation)]
        y = self.data['rmse'].values
        labels = PlotLabels(
            titles=f"Model Training RMSE / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of trained model",
            plot_type="rmse-learnable_params"
        )
        save_name = self.settings.generate_signature()
        Plot.box_plot(x, y, labels, self.exp_id, lambda x_coord: round(x_coord / 10_000.0) * 10_000, save_name)

    def plot_size_r2(self):
        function = self.settings.function
        size, depth = self.data['network_size'].values, self.data['depth'].values
        activation = self.data["activation_fn"].values
        x = [FNN(nodes=s, depth=d, activation=a).count_parameters() for (s, d, a) in zip(size, depth, activation)]
        y = self.data['r2'].values
        labels = PlotLabels(
            titles=f"Model Training R2 / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="R2 of trained model",
            plot_type="r2-learnable_params"
        )
        save_name = self.settings.generate_signature()
        Plot.box_plot(x, y, labels, self.exp_id, lambda x_coord: round(x_coord / 10_000.0) * 10_000, save_name)

    def plot_rmse_optimisation_time(self):
        function = self.settings.function
        activation = self.data["activation_fn"].values[0]
        [base, alt] = solvable_by(activation)
        x = self.data['rmse'].values
        y1 = self.data['optimisation_time_base'].values
        y2 = self.data['optimisation_time_alt'].values
        labels = PlotLabels(
            titles=[
                f"Model Training RMSE / Optimisation Time [{function.capitalize()}] [{base}]",
                f"Model Training RMSE / Optimisation Time [{function.capitalize()}] [{alt}]"
            ],
            x_axis="RMSE of trained model",
            y_axis="Optimisation Time (in seconds)",
            plot_type="rmse-opt_time"
        )
        save_name = self.settings.generate_signature()
        Plot.dual_scatter_plot(x, y1, y2, labels, self.exp_id, save_name)

    def plot_rmse_optimisation_error(self):
        function = self.settings.function
        activation = self.data["activation_fn"].values[0]
        [base, alt] = solvable_by(activation)
        x = self.data['rmse'].values
        y1 = self.data['optimisation_error_base'].values
        y2 = self.data['optimisation_error_alt'].values
        labels = PlotLabels(
            titles=[
                f"Model Training RMSE / Optimisation Error [{function.capitalize()}] [{base}]",
                f"Model Training RMSE / Optimisation Error [{function.capitalize()}] [{alt}]"
            ],
            x_axis="RMSE of trained model",
            y_axis="RMSE of Optimisation result",
            plot_type="rmse-opt_error"
        )
        save_name = self.settings.generate_signature()
        Plot.dual_scatter_plot(x, y1, y2, labels, self.exp_id, save_name)

    def plot_r2_optimisation_error(self):
        function = self.settings.function
        activation = self.data["activation_fn"].values[0]
        [base, alt] = solvable_by(activation)
        x = self.data['r2'].values
        y1 = self.data['optimisation_error_base'].values
        y2 = self.data['optimisation_error_alt'].values
        labels = PlotLabels(
            titles=[
                f"Model Training R2 / Optimisation Error [{function.capitalize()}] [{base}]",
                f"Model Training R2 / Optimisation Error [{function.capitalize()}] [{alt}]"
            ],
            x_axis="R2 of trained model",
            y_axis="RMSE of Optimisation result",
            plot_type="r2-opt_error"
        )
        save_name = self.settings.generate_signature()
        Plot.dual_scatter_plot(x, y1, y2, labels, self.exp_id, save_name)

    @staticmethod
    def plot_surface(fn: Function2D, x_max=5.12, n_points=100) -> None:
        r_min, r_max = -x_max, x_max
        spacing = 2 * x_max / n_points
        xaxis = np.arange(r_min, r_max, spacing)
        yaxis = np.arange(r_min, r_max, spacing)
        x, y = np.meshgrid(xaxis, yaxis)
        results: np.ndarray = fn(x, y)
        figure = plt.figure()
        axis = figure.gca(projection='3d')
        axis.plot_surface(x, y, results, cmap='jet', shade="false")
        plt.show()
        # plt.contour(x, y, results)
        # plt.show()
        # plt.scatter(x, y, results)
        # plt.show()


def plot_all(params: Dict, models: List[NeuralModel], exp_id: str):
    Plot.plot_surface(fn=functions.ackley, x_max=32.768)
    Plot.plot_surface(fn=functions.rosenbrock, x_max=32.768)
    Plot.plot_surface(fn=functions.rastrigin)
    Plot.plot_surface(fn=functions.sum_squares)

    settings = [from_dict(PlotSettings, it) for it in GridSearch.get_perm_sequence(list(params.items()))]
    for setting in settings:
        df = Plot.load_data(
            models,
            setting.function,
            lambda it: it.neural_config.activation_fn == setting.activation and it.neural_config.depth == setting.depth
        )
        # df = df[(np.abs(stats.zscore(df["optimisation_time_base"])) < 2)]
        plot = Plot(exp_id, setting, df)
        plot.plot_size_rmse()
        plot.plot_r2_optimisation_error()
        plot.plot_rmse_optimisation_time()
        plot.plot_rmse_optimisation_error()
        plot.plot_size_r2()
        plot.plot_size_rmse()
        plot.plot_size_optimisation_time()
        plot.plot_size_opt_error()
        plot.plot_size_opt_obj_error()


def compare_histograms(models, exp_id):
    least_error_id = "62fcf647cc4b54de3c2efdd4"
    most_error_id = "62fcf647cc4b54de3c2efe1c"

    least_error = [it for it in models if it.id == least_error_id][0]
    most_error = [it for it in models if it.id == most_error_id][0]

    Plot.weights_histogram(least_error, exp_id, save_name="least")
    Plot.weights_histogram(most_error, exp_id, save_name="most")


if __name__ == '__main__':
    l = []
    exp_id_list = ["batch-1-reg", "batch-2"]
    repo = NeuralModelRepository("mongodb://localhost:27017")
    for it in exp_id_list:
        models = repo.get_all(it)
        params = {
            "function": ["ackley", "rosenbrock", "rastrigin", "sum_squares"],
            "depth": [2, 4],
            "activation": ["ReLU", "Sigmoid"]
        }
        plot_all(params, models, it)
        #compare_histograms(models, it)
    with open(f"{ROOT_DIR}/resources/appendix.txt", "w") as file:
        file.writelines(l)
