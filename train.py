from tensor import Tensor
from model import Model
from loss_functions import Loss, TSE
from optimizers import Optimizer, SGD
from data import DataIterator, BatchIterator
import time

def train(model: Model,
          inputs: Tensor,
          targets: Tensor,
          epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = TSE(),
          optimizer: Optimizer = SGD()) -> None:
    tic = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = model.forward(batch.inputs) # Compute y^
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets) # Compute da[l]
            model.backward(grad) # Use da[l] to get dW[l-1], db[l-1], dW[l-2] etc.
            optimizer.step(model) # Update weights and biases according to the previously calculated gradients

        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
    toc = time.time()
    print("")
    print(f"[ FINISHED TRAINING IN: {round(toc-tic, 2)} SECONDS ]")
    print("")