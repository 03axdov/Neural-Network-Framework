from tensor import Tensor
from model import Model
from loss_function import Loss, TSE
from optimizer import Optimizer, SGD
from data import DataIterator, BatchIterator

def train(model: Model,
          inputs: Tensor,
          targets: Tensor,
          epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = TSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = model.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            model.backward(grad)
            optimizer.step(model)

        print(f"Epoch: {epoch}, Loss: {epoch_loss}")