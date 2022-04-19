import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

BOARD_COUNT = 100000
NEURONS = 64


def generate_board() -> dict:
    as_dict = dict()
    as_dict["score"] = 0
    for r in range(4):
        for c in range(4):
            tile = random.randint(0, 16)
            as_dict[f"{r},{c}"] = tile
            as_dict["score"] += 0 if tile <= 1 else (tile - 1) * (1 << tile)
    return as_dict


print("Starting board generation")
boards = np.array([generate_board() for _ in range(BOARD_COUNT)])
print(f"Generated {BOARD_COUNT} boards")

dataset = pd.DataFrame.from_records(boards)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('score')
test_labels = test_features.pop('score')

print("Separated dataset into training and testing datasets")

model = keras.Sequential([
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(1)
])
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.001))
print("Model compiled")
model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=100)
print("Model fitted")

evaluation = model.evaluate(test_features, test_labels)
print("Evaluation:", evaluation)

test_predictions = model.predict(test_features).flatten()

plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
max_score = max(boards, key=lambda b: b["score"])["score"]
lims = [0, max_score]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

difference = test_predictions - test_labels
plt.hist(difference, bins=25)
plt.xlabel('Prediction Difference')
plt.ylabel('Count')
plt.show()

quotient = test_predictions / test_labels
plt.hist(quotient, bins=25)
plt.xlabel('Prediction Quotient')
plt.ylabel('Count')
plt.show()

name = input('Enter model name or "q" to abort.').strip()
if name != "q":
    model.save(name)
