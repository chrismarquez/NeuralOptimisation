from torch import nn, optim


class Trainer:

    def __init__(self, net: nn.Module):
        loss_fn = nn.MSELoss()
        learning_rate = 10E-3
        optimiser = optim.SGD(net.parameters(), lr=learning_rate)
