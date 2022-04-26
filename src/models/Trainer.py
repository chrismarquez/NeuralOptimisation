import gc
import os
from decimal import Decimal
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import trange
from torch import nn, optim

from src.data.Dataset import Dataset
from src.models.FNN import FNN

Batch = Tuple[torch.Tensor, torch.Tensor]


class Trainer:

    def __init__(self, net: nn.Module, loss_fn=nn.MSELoss(), lr=1E-4, batch_size=128):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net = net.to(self._device).double()
        self._loss_fn = loss_fn
        self._lr = lr
        self._batch_size = batch_size
        self._optimiser = optim.SGD(net.parameters(), lr=lr)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, details: str = "") -> nn.Module:
        gc.collect()
        torch.cuda.empty_cache()  # Clean GPU memory before use
        x_train, y_train = torch.tensor(x_train).to(self._device), torch.tensor(y_train).to(self._device)
        batches = self._prepare_batches(x_train, y_train)
        # print(f"\tEpochs: {epochs}, Batch Size: {self._batch_size}, LR: {self._lr}\n")
        with trange(epochs, unit="epoch", leave=True) as progress:
            lr = "%.2E" % Decimal(self._lr)
            progress.set_description(f"Training Model: LR [{lr}] BS [{self._batch_size}] {details}")
            for _ in progress:
                running_loss = self._train_epoch(batches)
                progress.set_postfix(loss=running_loss)
        return self._net

    def save(self, folder: str, name: str) -> None:
        self._net.eval()
        path = f"./trained/{folder}/"
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self._net.state_dict(), f"{path}/{name}.pt")

    def _train_epoch(self, batches: List[Batch]) -> float:
        running_loss = 0.0
        for input_set, target_set in batches:
            running_loss += self._train_batch(input_set, target_set)
        return running_loss

    def _train_batch(self, input_set: torch.Tensor, target_set: torch.Tensor):
        self._net.zero_grad()
        output_set = self._net(input_set)
        loss = self._loss_fn(output_set, target_set)
        loss.backward()
        self._optimiser.step()
        return np.sqrt(loss.item())

    def _prepare_batches(self, input_dataset: torch.Tensor, target_dataset: torch.Tensor) -> List[Batch]:
        batch_count = len(target_dataset) // self._batch_size
        batches: List[Batch] = [
            (self._get_chunk(input_dataset, i), self._get_chunk(target_dataset, i))
            for i in range(batch_count)
        ]
        if len(target_dataset) % self._batch_size != 0:  # Remaining cluster
            last_input = input_dataset[batch_count * self._batch_size:, :]
            last_target = target_dataset[batch_count * self._batch_size:, :]
            batches.append((last_input, last_target))
        return batches

    def _get_chunk(self, dataset: torch.Tensor, index: int) -> torch.Tensor:
        start = index * self._batch_size
        end = (index + 1) * self._batch_size
        return dataset[start:end, :]


if __name__ == '__main__':
    trainer = Trainer(FNN(10, 2, "ReLU"))
    raw_dataset = np.loadtxt(f"samples/sum_squares.csv", delimiter=",")
    dataset = Dataset.create(raw_dataset)
    x_train, y_train = dataset.train
    trained_net = trainer.train(x_train, y_train, epochs=400)
    trainer.save("test", "sum_squares")
