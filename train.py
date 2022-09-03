from tensor import Tensor
from model import Model
from loss_functions import Loss, TSE
from optimizers import Optimizer, SGD
from data import DataIterator, BatchIterator
import time
import sys

def train(model: Model,
          inputs: Tensor,
          targets: Tensor,
          epochs: int = 5000,
          loss: Loss = TSE(batch_size=32),
          iterator: DataIterator = BatchIterator(batch_size=32),
          optimizer: Optimizer = SGD()) -> None:
    tic = time.time()
    for epoch in range(epochs):
        cost = 0.0
        for batch in iterator(inputs, targets):
            predicted = model.forward(batch.inputs) # Compute y^
            cost += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets) # Compute da[l]
            model.backward(grad) # Use da[l] to get dW[l-1], db[l-1], dW[l-2] etc.
            optimizer.step(model) # Update weights and biases according to the previously calculated gradients

        print(f"Epoch: {epoch + 1}, Loss: {cost}")
    toc = time.time()
    print("")
    print(f"[ FINISHED TRAINING IN: {round(toc-tic, 2)} SECONDS ]")
    print("")