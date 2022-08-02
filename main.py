import numpy as np
import matplotlib.pyplot as plt

from layers import Dense
from activation_functions import *
from loss_functions import *
from model import Model

def create_data(instances, classes):    # Based on https://cs231n.github.io/neural-networks-case-study/ - Spiral Dataset

    np.random.seed(0)

    X = np.zeros((instances*classes, 2))
    y = np.zeros(instances*classes, dtype='uint8')

    for class_number in range(classes):
        ix = range(instances*class_number, instances*(class_number+1))
        r = np.linspace(0.0, 1, instances) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, instances) + np.random.rand(instances) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X,y

def main():

    X, y = create_data(100, 3)

    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    # plt.show()

    model = Model([
        Dense(2, 3, ReLU()), 
        Dense(3,3, Softmax())],
        loss_function=CategoricalCrossentropy(),)

    output = model.forward(X)[:5]
    print(f"Output: {output}")


if __name__ == '__main__':
    main()