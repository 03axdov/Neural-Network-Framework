import numpy as np

class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

class Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backward(self, inputs):
        return inputs * (1 - inputs)
        