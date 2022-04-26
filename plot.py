
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn

from functions import Function2D
from models.FNN import FNN


def load_data(function: str) -> pd.DataFrame:
    df_training = pd.read_csv(f"trained/metadata/{function}.csv", delimiter=",")
    df_optimisation = pd.read_csv(f"trained/optimisation/{function}.csv", delimiter=",")
    df = df_training.set_index('id').join(df_optimisation.set_index('id'))
    df = df[df["activation_fn"] == "ReLU"]
    return df


def plot_size_optimisation_time(function: str):
    data = load_data(function)
    size = data['network_size'].values
    depth = data['depth'].values
    x = [FNN(nodes=s, depth=d, activation_fn=lambda: nn.ReLU()).count_parameters() for (s, d) in zip(size, depth)]
    y = data['computation_time'].values
    z = {}
    for (size, time) in zip(x, y):
        params_class = round(size / 10_000.0) * 10_000
        if params_class not in z:
            z[params_class] = []
        if not np.isnan(time):
            z[params_class].append(time)
    axes, values = zip(*list(sorted(z.items())))
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.set_title(f"Optimisation Time / Trainable Parameters [{function.capitalize()}]")
    ax.set_xlabel('Number of Trainable Parameters')
    ax.set_ylabel('Time (in seconds)')
    xtickNames = plt.setp(ax, xticklabels=axes)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    ax.boxplot(values)
    figure.show()


def plot_size_error(function: str):
    data = load_data(function)
    size = data['network_size'].values
    depth = data['depth'].values
    x = [FNN(nodes=s, depth=d, activation_fn=lambda: nn.ReLU()).count_parameters() for (s, d) in zip(size, depth)]
    y = data['location_error'].values
    z = {}
    for (size, error) in zip(x, y):
        params_class = round(size / 10_000.0) * 10_000
        if params_class not in z:
            z[params_class] = []
        if not np.isnan(error):
            z[params_class].append(error)
    axes, values = zip(*list(sorted(z.items())))
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.set_title(f"Optimisation Error / Trainable Parameters [{function.capitalize()}]")
    ax.set_xlabel('Number of Trainable Parameters')
    ax.set_ylabel("RMSE of (x', y') compared to (x*, y*)")
    xtickNames = plt.setp(ax, xticklabels=axes)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    ax.boxplot(values)
    figure.show()


    # figure = plt.figure()
    # figure.suptitle(f"Optimisation Error / Trainable Parameters [{function.capitalize()}]", fontsize=16)
    # plt.scatter(x, y)
    # figure.show()


def plot_size_optimisation_error(function: str):
    data = load_data(function)
    size = data['network_size'].values
    depth = data['depth'].values
    x = [FNN(nodes=s, depth=d, activation_fn=lambda: nn.ReLU()).count_parameters() for (s, d) in zip(size, depth)]
    y = data['optimum_error'].values
    z = {}
    for (size, error) in zip(x, y):
        params_class = round(size / 10_000.0) * 10_000
        if params_class not in z:
            z[params_class] = []
        if not np.isnan(error):
            z[params_class].append(error)
    axes, values = zip(*list(sorted(z.items())))
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.set_title(f"Optimum Value Error / Trainable Parameters [{function.capitalize()}]")
    ax.set_xlabel('Number of Trainable Parameters')
    ax.set_ylabel("RMSE of f' compared to f*")
    xtickNames = plt.setp(ax, xticklabels=axes)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    ax.boxplot(values)
    figure.show()

def plot_size_rmse(function: str):
    data = load_data(function)
    size = data['network_size'].values
    depth = data['depth'].values
    x = [FNN(nodes=s, depth=d, activation_fn=lambda: nn.ReLU()).count_parameters() for (s, d) in zip(size, depth)]
    y = data['rmse'].values
    z = {}
    for (size, error) in zip(x, y):
        params_class = round(size / 10_000.0) * 10_000
        if params_class not in z:
            z[params_class] = []
        if not np.isnan(error):
            z[params_class].append(error)
    axes, values = zip(*list(sorted(z.items())))
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.set_title(f"Model Training RMSE / Trainable Parameters [{function.capitalize()}]")
    ax.set_xlabel('Number of Trainable Parameters')
    ax.set_ylabel("RMSE of f' compared to f*")
    xtickNames = plt.setp(ax, xticklabels=axes)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    ax.boxplot(values)
    figure.show()


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
    #plot_surface(fn=functions.ackley, x_max=32.768)
    #plot_surface(fn=functions.rosenbrock, x_max=32.768)
    #plot_surface(fn=functions.rastrigin)
    #plot_surface(fn=functions.sum_squares)
    for fn in ["ackley", "rosenbrock", "rastrigin", "sum_squares"]:
        plot_size_optimisation_time(fn)
        plot_size_rmse(fn)
        plot_size_error(fn)
        plot_size_optimisation_error(fn)
