import numpy as np
from tensor import Tensor
import sys

class Loss: # Effectively Cost Functions as they apply to batches
    def loss(self, predicted: Tensor, actual: Tensor) -> float: # Calculate loss
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor: # Get da[l] from y^ - The derivative of y^ = da[l]
        raise NotImplementedError


class TSE(Loss):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return np.sum((predicted - actual) ** 2) / self.batch_size

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return 2 * (predicted - actual)


class CategoricalCrossentropy(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        pass

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        pass


class BinaryCrossentropy(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return (-predicted/actual) + (1-predicted) / (1-actual)


class Logistic(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return (-predicted/actual) + (1-predicted) / (1-actual)