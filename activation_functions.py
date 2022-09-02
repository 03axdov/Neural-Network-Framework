from layers import Layer
from tensor import Tensor
from typing import Callable
import numpy as np


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    def __init__(self, f: F, f_prime: F):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs    # Cache Z[l]
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad # dZ[l] = dA[l] * g[l]'(Z[l])


class Tanh(Activation):

    def tanh(self, x: Tensor) -> Tensor:
        return np.tanh(x)

    def tanh_prime(self, x: Tensor) -> Tensor:
        y = self.tanh(x)
        return 1 - y**2

    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)


class ReLU(Activation):
    def relu(self, x: Tensor) -> Tensor:
        return np.maximum(x, 0)

    def relu_prime(self, x: Tensor) -> Tensor:
        return (x > 0) * 1

    def __init__(self):
        super().__init__(self.relu, self.relu_prime)


class Softmax(Activation):
    def softmax(self, x: Tensor) -> Tensor:
        return

    def softmax_prime(self, x: Tensor) -> Tensor:
        return

    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)


class Sigmoid(Activation):
    def sigmoid(self, x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x: Tensor) -> Tensor:
        return np.exp(-x) / (1 + np.exp(-x))**2

    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)