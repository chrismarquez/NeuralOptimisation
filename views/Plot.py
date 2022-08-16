from typing import Callable, List, Optional, Literal, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.functions import Function2D
from models.FNN import FNN
from optimisation.Solver import solvable_by
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.db_models import NeuralModel, OptimisationProperties

Transform = Callable[[float], float]
Filter = Callable[[NeuralModel], bool]


class PlotLabels:

    def __init__(self, titles: Union[str, List[str]], x_axis: str, y_axis: str):
        self.titles = titles
        self.x_axis = x_axis
        self.y_axis = y_axis


class Plot:

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
                "r2": model.neural_properties.r2,
                "rmse": model.neural_properties.rmse,
                "optimisation_time_base": Plot._get_optimisation(model, "base").computation_time,
                "optimisation_error_base": Plot._get_optimisation(model, "base").location_error,
                "optimisation_time_alt": Plot._get_optimisation(model, "alt").computation_time,
                "optimisation_error_alt": Plot._get_optimisation(model, "alt").location_error,
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
        df_training = pd.read_csv(f"../../resources/trained/metadata/{function}.csv", delimiter=",")
        df_optimisation = pd.read_csv(f"../../resources/trained/optimisation/{function}.csv", delimiter=",")
        df = df_training.set_index('id').join(df_optimisation.set_index('id'))
        df = df[df["activation_fn"] == "ReLU"]
        return df

    @staticmethod
    def box_plot(x: List, y: List, labels: PlotLabels, x_transform: Transform = lambda x: x):
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
        figure.show()
        #figure.savefig("../resources/plots/test.pdf", dpi=600)


    @staticmethod
    def dual_scatter_plot(x: List, y1: List, y2: List, labels: PlotLabels):
        m1, b1 = np.polyfit(x, y1, 1)
        m2, b2 = np.polyfit(x, y2, 1)
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

        title1, title2 = labels.titles

        ax[0].set_axisbelow(True)
        plt.xscale("log")
        plt.yscale("log")
        ax[0].set_title(title1)
        ax[0].set_xlabel(labels.x_axis)
        ax[0].set_ylabel(labels.y_axis)
        ax[0].scatter(x, y1)
        ax[1].set_axisbelow(True)
        ax[1].set_title(title2)
        ax[1].set_xlabel(labels.x_axis)
        ax[1].set_ylabel(labels.y_axis)
        ax[1].scatter(x, y2)
        ax[0].plot(x, m1 * x + b1, color='red', rasterized=True)
        ax[1].plot(x, m2 * x + b2, color='red', rasterized=True)
        plt.tight_layout()
        figure.show()
        figure.savefig("../resources/plots/test.pdf", dpi=600)


    @staticmethod
    def plot_size_optimisation_time(function: str):
        data = Plot.load_data_from_file(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation=data["activation_fn"]).count_parameters() for (s, d) in zip(size, depth)]
        y = data['computation_time'].values
        labels = PlotLabels(
            titles=f"Optimisation Time / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="Time (in seconds)"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

    @staticmethod
    def plot_size_error(function: str):
        data = Plot.load_data_from_file(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['location_error'].values
        labels = PlotLabels(
            titles=f"Optimisation Error / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of (x', y') compared to (x*, y*)"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

        # figure = plt.figure()
        # figure.suptitle(f"Optimisation Error / Trainable Parameters [{function.capitalize()}]", fontsize=16)
        # plt.scatter(x, y)
        # figure.show()

    @staticmethod
    def plot_size_optimisation_error(function: str):
        data = Plot.load_data_from_file(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['optimum_error'].values
        labels = PlotLabels(
            titles=f"Optimum Value Error / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of f' compared to f*"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

    @staticmethod
    def plot_size_rmse(function: str, data: pd.DataFrame):
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['rmse'].values
        labels = PlotLabels(
            titles=f"Model Training RMSE / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of f' compared to f*"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

    @staticmethod
    def plot_size_r2(function: str, data: pd.DataFrame):
        size, depth = data['network_size'].values, data['depth'].values
        activation = data["activation_fn"].values
        x = [FNN(nodes=s, depth=d, activation=a).count_parameters() for (s, d, a) in zip(size, depth, activation)]
        y = data['r2'].values
        labels = PlotLabels(
            titles=f"Model Training R2 / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="R2 of trained model"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

    @staticmethod
    def plot_rmse_optimisation_time(function: str, data: pd.DataFrame):
        activation = data["activation_fn"].values[0]
        [base, alt] = solvable_by(activation)
        x = data['rmse'].values
        y1 = data['optimisation_time_base'].values
        y2 = data['optimisation_time_alt'].values
        labels = PlotLabels(
            titles=[
                f"Model Training RMSE / Optimisation Time [{function.capitalize()}] [{base}]",
                f"Model Training RMSE / Optimisation Time [{function.capitalize()}] [{alt}]"
            ],
            x_axis="RMSE of trained model",
            y_axis="Optimisation Time (in seconds)"
        )
        Plot.dual_scatter_plot(x, y1, y2, labels)

    @staticmethod
    def plot_rmse_optimisation_error(function: str, data: pd.DataFrame):
        activation = data["activation_fn"].values[0]
        [base, alt] = solvable_by(activation)
        x = data['rmse'].values
        y1 = data['optimisation_error_base'].values
        y2 = data['optimisation_error_alt'].values
        labels = PlotLabels(
            titles=[
                f"Model Training RMSE / Optimisation Error [{function.capitalize()}] [{base}]",
                f"Model Training RMSE / Optimisation Error [{function.capitalize()}] [{alt}]"
            ],
            x_axis="RMSE of trained model",
            y_axis="RMSE of Optimisation result"
        )
        Plot.dual_scatter_plot(x, y1, y2, labels)

    @staticmethod
    def plot_r2_optimisation_error(function: str, data: pd.DataFrame):
        activation = data["activation_fn"].values[0]
        [base, alt] = solvable_by(activation)
        x = data['r2'].values
        y1 = data['optimisation_error_base'].values
        y2 = data['optimisation_error_alt'].values
        labels = PlotLabels(
            titles=[
                f"Model Training R2 / Optimisation Error [{function.capitalize()}] [{base}]",
                f"Model Training R2 / Optimisation Error [{function.capitalize()}] [{alt}]"
            ],
            x_axis="R2 of trained model",
            y_axis="RMSE of Optimisation result"
        )
        Plot.dual_scatter_plot(x, y1, y2, labels)

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


if __name__ == '__main__':
    # plot_surface(fn=functions.ackley, x_max=32.768)
    # plot_surface(fn=functions.rosenbrock, x_max=32.768)
    # plot_surface(fn=functions.rastrigin)
    # plot_surface(fn=functions.sum_squares)

    exp_id = "batch-2"
    repo = NeuralModelRepository("mongodb://localhost:27017")
    models = repo.get_all(exp_id)
    for fn in ["ackley", "rosenbrock", "rastrigin", "sum_squares"]:
        for d in [2, 4]:
            df = Plot.load_data(models, fn,
                                lambda it: it.neural_config.activation_fn == "ReLU" and it.neural_config.depth == d)

            #df = df[(np.abs(stats.zscore(df["optimisation_time_base"])) < 2)]
            #Plot.plot_size_r2(fn, df)
            #Plot.plot_size_rmse(fn, df)
            Plot.plot_rmse_optimisation_time(fn, df)
    #      Plot.plot_size_optimisation_time(fn)
    #      Plot.plot_size_rmse(fn)
    #      Plot.plot_size_error(fn)
    #      Plot.plot_size_optimisation_error(fn)




    # All comparisons must be made separated by depth and Activation Function
    # Depth affects the slope and intercept of the tendency lines, but the proportionality appears to be kept as linear
    # Act Function affects the learned function, apparently some functions were learned better (or with more stability)
    #   with a specific activation.

    # TODO: Add Colored Line plots, several lines indicating alike configs differing only by x-axis component
    #  (i.e. learnable params)
