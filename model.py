from tensor import Tensor
from layers import Layer
from typing import Sequence, Iterator, Tuple


class Model:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers # Activation functions count as layers

    def forward(self, inputs: Tensor) -> Tensor: # Calculate y^
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor: # Calculate gradients
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]: # Used by the optimizer to update params accordingly
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad