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
        self.params["w"] = np.random.randn(input_size, output_size) * 0.01
        self.params["b"] = np.zeros(output_size)
        self.input_size = input_size

    def forward(self, inputs:Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T
