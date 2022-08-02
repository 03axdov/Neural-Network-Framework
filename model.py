import numpy as np

class Model():
    def backward(self, X, y, output, layers, loss):
        output_layer = layers[-1]
        output_layer.backward()

        self.output_error = y - output