import torch.nn as nn
from config import LossName


def get_loss(name: LossName):
    if name == LossName.MSE:
        return nn.MSELoss()
    if name == LossName.BCE:
        return nn.BCELoss()
    raise ValueError(name)
