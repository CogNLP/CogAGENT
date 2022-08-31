import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def loss(self, batch, loss_function):
        pass

    def forward(self, batch):
        pass

    def evaluate(self, batch, metric_function):
        pass

    def predict(self, batch):
        pass