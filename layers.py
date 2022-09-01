import numpy as np
from typing import Dict, Callable
from tensor import Tensor

class Layer:
    def __init__(self):
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        # Inputs: (batch_size, input_size)
        # Outputs: (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size) * 0.01  # Too large weights will result in calculation on 'flat' parts of certain activation functions such as tanh
        self.params["b"] = np.zeros(output_size)

    def forward(self, inputs:Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    def __init__(self, f: F, f_prime: F):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad


# Tanh

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


# ReLU

def relu(x: Tensor) -> Tensor:
    return np.maximum(x, 0)

def relu_prime(x: Tensor) -> Tensor:
    return (x > 0) * 1

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)