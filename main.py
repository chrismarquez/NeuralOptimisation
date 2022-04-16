import numpy as np
import torch

import functions
from dataset import Dataset
from models.FNN import FNN
from models.Regressor import Regressor
from models.Trainer import Trainer


def train(dataset):
    trainer = Trainer(FNN())
    x_train, y_train = dataset.train
    trained_net = trainer.train(x_train, y_train, 400)
    trainer.save("sum_squares")
    return trained_net

def main():
    raw_dataset = np.loadtxt("samples/sum_squares.csv", delimiter=",")
    dataset = Dataset.create(raw_dataset)

    #trained_net = train(dataset)

    trained_net = FNN()
    trained_net.load_state_dict(torch.load("trained/sum_squares.pt"))

    predictor = Regressor(trained_net)
    x = np.array([[1.0, 2.0]])
    test = torch.tensor(x).double()

    out = predictor.predict(test)
    # real = functions.sum_squares(x[:, 0], x[:, 1])

    print(out)

    x_dev, y_dev = dataset.dev
    dev_error = predictor.evaluate(torch.tensor(x_dev).double(), torch.tensor(y_dev).double())
    print(dev_error)

    # print(real)


if __name__ == '__main__':
    main()
