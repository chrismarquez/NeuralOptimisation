from __future__ import annotations
from typing import Tuple

import numpy as np

Subset = Tuple[np.ndarray, np.ndarray]


class Dataset:

    def __init__(self, train: Subset, dev: Subset, test: Subset):
        self.train = train
        self.dev = dev
        self.test = test

    @staticmethod
    def create(dataset: np.ndarray) -> Dataset:
        size = len(dataset)
        split_points = [0.7, 0.2, 0.1]
        shuffled_dataset = Dataset.shuffle(dataset)
        subsets = []
        prev_i = 0
        for i in split_points:
            index = int(i * size) + prev_i
            subsets.append(shuffled_dataset[prev_i:index, :])
            prev_i = int(i * size)
        subsets = [Dataset.split(s) for s in subsets]
        train, dev, test = subsets
        return Dataset(train, dev, test)

    @staticmethod
    def split(dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        split = dataset.shape[1] - 1
        input_dataset = dataset[:, :split]
        target_dataset = dataset[:, split:]
        return input_dataset, target_dataset

    @staticmethod
    def shuffle(dataset: np.ndarray) -> np.ndarray:
        shuffled_indices = np.random.permutation(len(dataset))
        return dataset[shuffled_indices]
