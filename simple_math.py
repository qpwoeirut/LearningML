import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
LABELS = 200000
NEURONS = 2
LAYERS = 2
EPOCHS = 100  # the EarlyStopping callback will usually end the training well before this number
MIN_DELTA = 10
LEARNING_RATE = 0.001


def generate_label() -> dict:
    as_dict = dict()
    as_dict["a"] = random.randint(0, 30000)
    as_dict["b"] = random.randint(0, 30000)
    as_dict["res"] = as_dict["a"] + as_dict["b"]
    return as_dict


def show_loss_plot(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 1e6])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()


def show_actual_predicted_plot(test_predictions):
    plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    lims = [0, max(labels, key=lambda b: b["res"])["res"]]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()
    plt.clf()


def show_difference_plot(differences):
    plt.hist(differences, bins=25)
    plt.xlabel('Prediction Difference')
    plt.ylabel('Count')
    plt.show()
    plt.clf()


print("Starting label generation")
labels = np.array([generate_label() for _ in range(LABELS)])
print(f"Generated {LABELS} labels")

dataset = pd.DataFrame.from_records(labels)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('res')
test_labels = test_features.pop('res')

print("Separated dataset into training and testing datasets")

model_layers = [layers.Dense(NEURONS, activation='relu') for _ in range(LAYERS)] + [layers.Dense(1)]
stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=MIN_DELTA)
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
    callbacks=[stopper],
)
print("Model fitted")

evaluation = model.evaluate(test_features, test_labels)
print("Evaluation:", evaluation)

test_features = test_features.head(100)
test_labels = test_labels[:100]
test_predictions = model.predict(test_features).flatten()
for feature, label, prediction in zip(test_features.itertuples(), test_labels, test_predictions):
    print(feature.a, feature.b, label, prediction, prediction - label)

show_difference_plot(test_predictions - test_labels)
show_loss_plot(history)
show_actual_predicted_plot(test_predictions)