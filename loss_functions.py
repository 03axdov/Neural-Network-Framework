import numpy as np
from tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float: # Calculate loss
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor: # Get da[l] from y^ - The derivative of y^ = da[l]
        raise NotImplementedError


class TSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class CategoricalCrossentropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        pass

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        pass


class BinaryCrossentropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return (-predicted/actual) + (1-predicted) / (1-actual)


class Logistic(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return (-predicted/actual) + (1-predicted) / (1-actual)