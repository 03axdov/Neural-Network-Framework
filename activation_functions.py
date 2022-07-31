import numpy as np

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs- np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        