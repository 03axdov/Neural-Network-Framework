import numpy as np

from train import train
from model import Model
from layers import Dense, Tanh
from typing import List
from optimizer import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0,0,0,1]
    elif x % 5 == 0:
        return [0,0,1,0]
    elif x % 3 == 0:
        return [0,1,0,0]
    else:
        return [1,0,0,0]

def binary_encode(x:int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

inputs = np.array([binary_encode(x) for x in range(101, 1024)])

targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])

model = Model([
    Dense(input_size=10, output_size=50),
    Tanh(),
    Dense(input_size=50, output_size=4)
])

train(model, inputs, targets, epochs=50, optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = model.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])