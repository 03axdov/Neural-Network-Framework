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
        self.params["w"] = np.random.randn(input_size, output_size) * 0.01  # Initialize weights
        self.params["b"] = np.zeros(output_size)    # Initialize biases
        self.inputs = np.array([])

    def forward(self, inputs:Tensor) -> Tensor:
        self.inputs = inputs    # Cache a[l-1]
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor: # dZ[l] = dA[l] * g[l]'(Z[l]) --> See activation_functions.py
        self.grads["b"] = np.sum(grad, axis=0)  # Bias gradients - np.sum(dZ[l], axis=0, keepdims=True) - For another implementation - Biases: column vector instead of row vector
        self.grads["w"] = self.inputs.T @ grad  # dW[l] = (grad -->) dZ[l] * A[l-1].T (<-- self.inputs.T) - Could divide the result by m
        return grad @ self.params["w"].T
