import numpy as np
from sklearn import metrics
import torch
from torch import nn

from data.Dataset import Dataset
from models.FNN import FNN


class Regressor:

    def __init__(self, net: nn.Module):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net = net.to(self._device).double().eval()
        
    def predict(self, x_predict: torch.tensor) -> np.ndarray:
        with torch.no_grad():
            output: torch.Tensor = self._net(x_predict.to(self._device))
            return output.detach().cpu().numpy()

    def evaluate(self, x_test: torch.Tensor, y_target: torch.Tensor) -> (float, float):
        y_predicted = self.predict(x_test)
        y_target = y_target.detach().cpu().numpy()
        error = metrics.mean_squared_error(y_target, y_predicted, squared=False)
        r2 = metrics.r2_score(y_target, y_predicted)
        return error, r2


if __name__ == '__main__':

    raw_dataset = np.loadtxt(f"samples/sum_squares.csv", delimiter=",")
    dataset = Dataset.create(raw_dataset)
    trained_net = FNN(10, 2, "ReLU").load("trained/test/sum_squares.pt")

    predictor = Regressor(trained_net)
    x = np.array([[1.0, 2.0]])
    test = torch.tensor(x).double()

    out = predictor.predict(test)
    # real = functions.sum_squares(x[:, 0], x[:, 1])

    print(out)

    x_dev, y_dev = dataset.dev
    dev_error = predictor.evaluate(torch.tensor(x_dev).double(), torch.tensor(y_dev).double())
    print(dev_error)
