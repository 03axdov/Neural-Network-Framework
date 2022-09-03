import numpy as np

from train import train
from model import Model
from layers import Dense
from activation_functions import *
from typing import List
from optimizers import SGD
from loss_functions import *
from data import *

import tensorflow as tf

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    print(f"[ X_TRAIN : {x_train.shape} ]")

    model = Model([
        Dense(input_size=13, output_size=50),
        ReLU(),
        Dense(input_size=50, output_size=50),
        ReLU(),
        Dense(input_size=50, output_size=1),
    ])

    train(model, x_train, y_train, epochs=50, iterator=BatchIterator(batch_size=64), optimizer=SGD(lr=0.001))

    for x, y in zip(x_test, y_test):
        predicted = model.forward([x])
        print(f"[ PREDICTED : {predicted} ]")
        print(f"[ ACTUAL : {y} ]")


if __name__ == "__main__":
    main()