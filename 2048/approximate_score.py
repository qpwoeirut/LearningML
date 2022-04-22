import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers

BOARD_COUNT = 100000
NEURONS = 64
LAYERS = 6
EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_IDENTIFIER = f"{LAYERS}-{NEURONS}-{EPOCHS}-{LEARNING_RATE}"

pathlib.Path(MODEL_IDENTIFIER).mkdir(exist_ok=True)


def generate_board() -> dict:
    as_dict = dict()
    as_dict["score"] = 0
    for r in range(4):
        for c in range(4):
            tile = random.randint(0, 16)
            as_dict[f"{r},{c}"] = tile
            as_dict["score"] += 0 if tile <= 1 else (tile - 1) * (1 << tile)
    return as_dict


def save_loss_plot(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, max(history.history["loss"])])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(pathlib.Path(MODEL_IDENTIFIER) / "loss")
    plt.clf()


def save_actual_predicted_plot(test_predictions):
    plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    max_score = max(boards, key=lambda b: b["score"])["score"]
    lims = [0, max_score]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig(pathlib.Path(MODEL_IDENTIFIER) / "actual_predicted")
    plt.clf()


def save_difference_plot(differences):
    plt.hist(differences, bins=25)
    plt.xlabel('Prediction Difference')
    plt.ylabel('Count')
    plt.savefig(pathlib.Path(MODEL_IDENTIFIER) / "difference")
    plt.clf()


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

model_layers = [layers.Dense(NEURONS, activation='relu') for _ in range(LAYERS)] + [layers.Dense(1)]
model = keras.Sequential(model_layers)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
print("Model compiled")
history = model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=EPOCHS)
print("Model fitted")

save_loss_plot(history)

evaluation = model.evaluate(test_features, test_labels)
print("Evaluation:", evaluation)

test_predictions = model.predict(test_features).flatten()
save_actual_predicted_plot(test_predictions)

difference = test_predictions - test_labels
save_difference_plot(difference)

name = input('Hit enter to save or q to abort ').strip()
if name != "q":
    model.save("model_" + MODEL_IDENTIFIER)
