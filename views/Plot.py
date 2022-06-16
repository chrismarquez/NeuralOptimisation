from typing import Callable, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.functions import Function2D
from models.FNN import FNN

Transform = Callable[[float], float]


class PlotLabels:

    def __init__(self, title: str, x_axis: str, y_axis: str):
        self.title = title
        self.x_axis = x_axis
        self.y_axis = y_axis


class Plot:

    @staticmethod
    def load_data(function: str) -> pd.DataFrame:
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
        ax.set_title(labels.title)
        ax.set_xlabel(labels.x_axis)
        ax.set_ylabel(labels.y_axis)
        xtickNames = plt.setp(ax, xticklabels=axes)
        plt.setp(xtickNames, rotation=45, fontsize=8)
        ax.boxplot(values)
        figure.show()

    @staticmethod
    def plot_size_optimisation_time(function: str):
        data = Plot.load_data(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['computation_time'].values
        labels = PlotLabels(
            title=f"Optimisation Time / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="Time (in seconds)"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

    @staticmethod
    def plot_size_error(function: str):
        data = Plot.load_data(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['location_error'].values
        labels = PlotLabels(
            title=f"Optimisation Error / Trainable Parameters [{function.capitalize()}]",
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
        data = Plot.load_data(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['optimum_error'].values
        labels = PlotLabels(
            title=f"Optimum Value Error / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of f' compared to f*"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)

    @staticmethod
    def plot_size_rmse(function: str):
        data = Plot.load_data(function)
        size, depth = data['network_size'].values, data['depth'].values
        x = [FNN(nodes=s, depth=d, activation="ReLU").count_parameters() for (s, d) in zip(size, depth)]
        y = data['rmse'].values
        labels = PlotLabels(
            title=f"Model Training RMSE / Trainable Parameters [{function.capitalize()}]",
            x_axis="Number of Trainable Parameters",
            y_axis="RMSE of f' compared to f*"
        )
        Plot.box_plot(x, y, labels, lambda x_coord: round(x_coord / 10_000.0) * 10_000)


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
    for fn in ["ackley", "rosenbrock", "rastrigin", "sum_squares"]:
        Plot.plot_size_optimisation_time(fn)
        Plot.plot_size_rmse(fn)
        Plot.plot_size_error(fn)
        Plot.plot_size_optimisation_error(fn)
