import numpy as np

from activation_functions import ReLU, Softmax
from layers import Dense
from loss_functions import Loss, CategoricalCrossentropy

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

X, y = create_data(100, 3)

def main():
    dense1 = Dense(2, 3)
    activation1 = ReLU()

    dense2 = Dense(3, 3)
    activation2 = Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])

    loss_function = CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    print("Loss: ", loss)



if __name__ == '__main__':
    main()