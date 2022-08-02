import numpy as np

class Model():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current_output = inputs
        for t, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
            if t == len(self.layers) - 1:
                return current_output

    def backward(self, X, y, output, loss):
        output_layer = self.layers[-1]
        output_layer.backward()

        self.output_error = y - output