import numpy as np

class Model():
    def __init__(self, layers, learning_rate, loss_function):
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def forward(self, inputs):
        current_output = inputs
        for t, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
            if t == len(self.layers) - 1:
                return current_output

    def one_hot(self, Y):
        one_hot_y = np.zeros((Y.size, Y.max() + 1))
        one_hot_y[np.arange(Y.size), Y] = 1
        return one_hot_y.T

    def fit(self, X, y, epochs=100, displayUpdate=100):
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                y_pred = self.fit_partial(x, y, target)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0: # First epoch and the specified one
                loss = self.loss_function.forward(y_pred, y)
                print(f"[INFO] epoch={epoch + 1}, loss={loss}")
    
    def fit_partial(self, x, y, target):
        A = [np.atleast_2d(x)]
        
        for t, layer in enumerate(self.layers):
            out = layer.forward(A[t])
            A.append(out)

        error = A[-1] - y
        D = [error * layer.activation.backward(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(layer.weights.T)
            delta = delta * self.layer.activation.backward(A[layer])
            D.append(delta)

        D = D[::-1]
		# WEIGHT UPDATE PHASE
		# loop over the layers
        for t, layer in enumerate(self.layers):
			# update our weights by taking the dot product of the layer
			# activations with their respective deltas, then multiplying
			# this value by some small learning rate and adding to our
			# weight matrix -- this is where the actual "learning" takes
			# place
            layer.weights[t] += -self.learning_rate * A[t].T.dot(D[t])

        return out
