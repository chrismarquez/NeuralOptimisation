import numpy as np
from sklearn import metrics
import torch
from torch import nn


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
