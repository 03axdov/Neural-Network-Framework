from tensor import Tensor
from model import Model
import sys


class Optimizer:
    def step(self, model: Model) -> None: # Update params
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001, lambd = 0) -> None:   # learning rate = 0.001, lambd --> lambda for regularization
        self.lr = lr
        self.lambd = lambd

    def step(self, model: Model) -> None:
        for param, grad in model.params_and_grads():
            param -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def step(self, model: Model) -> None:
        pass