import numpy as np

class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, inputs):
        return inputs > 0

class Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.output

    def backward(self):
        softmax = np.reshape(self.output, (1, -1))

        d_softmax = (                                                           
        softmax * np.identity(softmax.size)                                 
        - softmax.transpose() @ softmax)

        return d_softmax

class Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backward(self, inputs):
        return inputs * (1 - inputs)
        