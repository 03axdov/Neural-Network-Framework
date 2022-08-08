from tensor import Tensor
from model import Model


class Optimizer:
    def step(self, model: Model) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, model: Model) -> None:
        for param, grad in model.params_and_grads():
            param -= self.lr * grad