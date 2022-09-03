import numpy as np
from tensor import Tensor

class Loss: # Effectively Cost Functions as they apply to batches
    def loss(self, predicted: Tensor, actual: Tensor) -> float: # Calculate loss
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor: # Get da[l] from y^ - The derivative of y^ = da[l]
        raise NotImplementedError


class TSE(Loss):
    def __init__(self, regularization_lambda=0):
        self.regularization_lambda = regularization_lambda

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return 2 * (predicted - actual) + self.regularization_lambda


class CategoricalCrossentropy(Loss):
    def __init__(self, regularization_lambda=0):
        self.regularization_lambda = regularization_lambda

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        pass

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        pass


class BinaryCrossentropy(Loss):
    def __init__(self, regularization_lambda=0):
        self.regularization_lambda = regularization_lambda

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return (-predicted/actual) + (1-predicted) / (1-actual) + self.regularization_lambda


class Logistic(Loss):
    def __init__(self, regularization_lambda=0):
        self.regularization_lambda = regularization_lambda

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: Tensor, actual: Tensor, weights: Tensor) -> Tensor:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return (-predicted/actual) + (1-predicted) / (1-actual) + self.regularization_lambda