from tensor import Tensor
from layers import Layer
from typing import Sequence, Iterator, Tuple


class Model:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers # Activation functions count as layers
        self.weights = []

    def forward(self, inputs: Tensor) -> Tensor: # Calculate y^
        self.weights = []    # As to prevent large matrixes between epochs
        print(f"[ INPUTS : {inputs.shape} ]")
        for layer in self.layers:
            inputs = layer.forward(inputs)
            try:    # W - The weight matrix will be used by the loss function for regularization
                self.weights.append(layer.params['w'])
            except KeyError:    # Is an activation layer without weights
                continue
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