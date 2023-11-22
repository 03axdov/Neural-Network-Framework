# Neural-Network-Framework
> A Neural Network framework, built with Python and Numpy. Currently working on another framework called Muskie, specifically for Computer Vision. This framework is quite prone to exploding gradients (i.e. gradients whose values move towards infinity, thus 'destroying' the network), so appropriate models should be constructed.

Some improvements that can be made:
- More optimizers, activation functions etc.
- More layers, such as Conv2D and other convolutional layers
- Training on GPU's
- Saving model parameters and initializing these later on
- Additional functionality for working with databases

```
pip install -r requirements.txt
```
Alternatively just pip install numpy as that is the only external package used. Note that Tensorflow is used in main.py, however only as a means of testing models on keras datasets.

Current syntax:
```python
import numpy as np
from train import train
from model import Model
from layers import *
from activation_functions import *
from optimizers import *
from loss import *
from data import *

inputs = np.array([<individual input>] * your input length)
targets = np.array([<one hot encoded targets>] * your input length)

model = Model([
  Dense(input_size=len(<individual input>), output_size=50) # The output size can be whatever you want
  ReLU(),
  ... # Hidden Layers
  Dense(input_size=50, output_size=<number of targets>)
])

# Train the model
train(model, inputs, targets, epochs=50, optimizer=SGD(lr=0.001), iterator=BatchIterator(batch_size=32), loss=TSE()) # Parameters can, of course, be customized

# Make predictions
predictions = model.forward(instance)
```

See 'activation_functions.py', 'layers.py' etc. to browse which algorithms are currently implemented.

The theory behind neural networks as well as certain naming conventions is derived from <a href="https://www.youtube.com/c/Deeplearningai">Andrew Ng</a>, co-founder and head of Google Brain and former chief scientist at Baidu.
