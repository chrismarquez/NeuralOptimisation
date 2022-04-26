import numpy as np
import torch

from dataset import Dataset
from models.FNN import FNN
from models.ModelsExecutor import ModelsExecutor
from models.Regressor import Regressor
from models.Trainer import Trainer
from optimisation.OptimisationExecutor import OptimisationExecutor


def train(dataset: Dataset):
    trainer = Trainer(FNN.instantiate())
    x_train, y_train = dataset.train
    trained_net = trainer.train(x_train, y_train, epochs=400)
    trainer.save("test", "sum_squares")
    return trained_net



def test(trained_net, dataset):
    predictor = Regressor(trained_net)
    x = np.array([[1.0, 2.0]])
    test = torch.tensor(x).double()

    out = predictor.predict(test)
    # real = functions.sum_squares(x[:, 0], x[:, 1])

    print(out)

    x_dev, y_dev = dataset.dev
    dev_error = predictor.evaluate(torch.tensor(x_dev).double(), torch.tensor(y_dev).double())
    print(dev_error)



if __name__ == '__main__':
    #train_all_models()
    # main()
    # raw_dataset = np.loadtxt(f"samples/sum_squares.csv", delimiter=",")
    # dataset = Dataset.create(raw_dataset)
    # train(dataset)
    #exec = OptimisationExecutor()
    #exec.run_all_jobs()
    exec = ModelsExecutor()
    exec.run_all_jobs()

