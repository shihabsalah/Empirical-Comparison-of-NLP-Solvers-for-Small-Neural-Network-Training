import torch.optim as optim
from config import OptimizerName


def get_optimizer(name: OptimizerName, params, lr: float):
    if name == OptimizerName.SGD:
        return optim.SGD(params, lr=lr)
    if name == OptimizerName.ADAM:
        return optim.Adam(params, lr=lr)
    if name == OptimizerName.LBFGS:
        return optim.LBFGS(params, lr=lr)
    raise ValueError(name)
