import gc
import os
import tempfile
from decimal import Decimal
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn, optim
from tqdm.auto import trange

from cluster.JobInit import init_container
from models.FNN import FNN

Batch = Tuple[torch.Tensor, torch.Tensor]


class NanException(RuntimeError):
    pass


class Trainer:

    def __init__(self, net: nn.Module, loss_fn=nn.MSELoss(), lr=1E-4, batch_size=128,
                 l1_reg_lambda: Optional[float] = None):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net = net.to(self._device).double()
        self._loss_fn = loss_fn
        self._lr = lr
        self._batch_size = batch_size
        self._optimiser = optim.SGD(net.parameters(), lr=lr)
        self.l1_reg_lambda = l1_reg_lambda

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, details: str = "") -> nn.Module:
        gc.collect()
        torch.cuda.empty_cache()  # Clean GPU memory before use
        x_train, y_train = torch.tensor(x_train).to(self._device), torch.tensor(y_train).to(self._device)
        batches = self._prepare_batches(x_train, y_train)
        # print(f"\tEpochs: {epochs}, Batch Size: {self._batch_size}, LR: {self._lr}\n")
        with trange(epochs, unit="epoch", leave=True) as progress:
            lr = "%.2E" % Decimal(self._lr)
            progress.set_description(
                f"Training Model: LR [{lr}] BS [{self._batch_size}] L1-Reg [{self.l1_reg_lambda}] {details}"
            )
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

    def get_model_data(self) -> bytes:
        self._net.eval()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
            torch.save(self._net.state_dict(), file)
            file.seek(0)
            x = file.read()
            return x

    def _train_epoch(self, batches: List[Batch]) -> float:
        running_loss = 0.0
        for input_set, target_set in batches:
            running_loss += self._train_batch(input_set, target_set)
        if np.isnan(running_loss):
            raise NanException()
        return running_loss

    def _train_batch(self, input_set: torch.Tensor, target_set: torch.Tensor):
        self._net.zero_grad()
        output_set = self._net(input_set)
        loss = self._loss_fn(output_set, target_set)
        if self.l1_reg_lambda is not None:
            loss += self._get_penalty()
        loss.backward()
        self._optimiser.step()
        return np.sqrt(loss.item())

    def _get_penalty(self) -> torch.Tensor:
        reg_loss = sum(torch.linalg.norm(param, 1) for param in self._net.parameters())
        return self.l1_reg_lambda * reg_loss

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
    trainer = Trainer(FNN(420, 2, "Sigmoid"))
    container = init_container()
    sample_repo = container.sample_repository()
    dataset = sample_repo.get("62dcc587ce0f41019d2d7d78").to_dataset()
    x_train, y_train = dataset.train
    trained_net = trainer.train(x_train, y_train, epochs=500)
