import numpy as np

from train import train
from model import Model
from layers import Dense
from activation_functions import Tanh, ReLU
from typing import List
from optimizers import SGD

def main():
    def fizz_buzz_encode(x: int) -> List[int]:
        if x % 15 == 0:
            return [0,0,0,1]
        elif x % 5 == 0:
            return [0,0,1,0]
        elif x % 3 == 0:
            return [0,1,0,0]
        else:
            return [1,0,0,0]

    inputs = np.array([[x] for x in range(101, 1024)])

    targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])

    model = Model([
        Dense(input_size=1, output_size=50),
        ReLU(),
        Dense(input_size=50, output_size=4)
    ])

    train(model, inputs, targets, epochs=50, optimizer=SGD(lr=0.001))

    accuracy = 0
    for x in range(1, 101):
        predicted = model.forward([x])
        predicted_idx = np.argmax(predicted)
        actual_idx = np.argmax(fizz_buzz_encode(x))
        if predicted_idx == actual_idx: accuracy += 1
        labels = [str(x), "fizz", "buzz", "fizzbuzz"]
        print(x, labels[predicted_idx], labels[actual_idx], "ACCURACY: ", round(accuracy/x, 4))


if __name__ == "__main__":
    main()