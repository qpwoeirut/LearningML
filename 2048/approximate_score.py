import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
BOARD_COUNT = 500000
NEURONS = 20
LAYERS = 2
EPOCHS = 500
LEARNING_RATE = 0.001

MODEL_IDENTIFIER = f"{LAYERS}-{NEURONS}-{EPOCHS}-{LEARNING_RATE}"


def generate_board() -> dict:
    board = [random.randint(0, 13) for _ in range(16)]
    as_dict = dict()
    as_dict["score"] = 0
    for pos in range(16):
        as_dict[f"a_{pos}"] = 0 if board[pos] == 0 else 1 << board[pos]
        as_dict["score"] += 0 if board[pos] <= 1 else (board[pos] - 1) * (1 << board[pos])
        if pos != 3 and pos != 7 and pos != 11 and pos != 15:
            as_dict[f"eq_{pos},{pos+1}"] = 1 if board[pos] == board[pos + 1] else 0
        if pos < 12:
            as_dict[f"eq_{pos},{pos+4}"] = 1 if board[pos] == board[pos + 4] else 0
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
    max_score = min(max_score, 8e7)
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
stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=10000)
model = keras.Sequential(model_layers)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
print("Model compiled")
history = model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=EPOCHS,
    callbacks=[stopper]
)
print("Model fitted")

evaluation = model.evaluate(test_features, test_labels)
print("Evaluation:", evaluation)

name = input('Hit enter to save or q to abort ').strip()
if name != "q":
    pathlib.Path(MODEL_IDENTIFIER).mkdir(exist_ok=True)
    save_loss_plot(history)

    test_predictions = model.predict(test_features).flatten()
    save_actual_predicted_plot(test_predictions)

    difference = test_predictions - test_labels
    save_difference_plot(difference)
    model.save("model_" + MODEL_IDENTIFIER)
