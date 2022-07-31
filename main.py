import numpy as np
from activation_functions import ReLU, Softmax
from layers import Dense

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
    pass


if __name__ == '__main__':
    main()