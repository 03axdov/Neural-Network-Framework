# Neural-Network-Framework
A Neural Network framework, built with Python and Numpy.

While the framework is currently functional it is not done. There are several improvements that can be made:
- More optimizers, activation functions etc.
- More layers, such as Conv2D and other convolutional layers
- Training on GPU's
- Saving model parameters and initializing these later on
- Additional functionality for working with databases

If you want to run the repository, run "pip install -r requirements.txt" and write your code in main.py. An example of a model built for the fizzbuzz problem is currently given. If a number is divisible by 3 it should return 'fizz', if the number is divisble by 5 it should return 'buzz' and if it's divisble by 15, 'fizzbuzz'.

Current syntax:
```python
import numpy as np
from train import train
from model import Model
from layers import *
from activation_functions import *
from optimizers import *

inputs = np.array([<individual input length>] * your input length)
targets = np.array([<one hot encoded targets>] * your input length)

model = Model([
  Dense(input_size=<individual input length>, output_size=50) # The output size can be whatever you want
  ReLU(),
  ... # Hidden Layers
  Dense(input_size=50, output_size=<number of targets>)
])

# Train the model
train(model, inputs, targets, epochs=50, optimizer=SGD(lr=0.001)) # The epochs and optimizer can be whatever

# Make predictions
predictions = model.forward(instance)
```
