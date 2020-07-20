#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from preprocess.format import read_dataset, split_data, segments_2d
from learn import categorize_chips
from learn.models import rfmlcnn

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

PATH = "../data/dataset/1"

# TODO Function to return metadata
# The IDs of tags of a given model
NTAG213 = (1, 2, 3, 4, 5)
MIFARE = (6, 7, 8)
FELICA = (9,)


def svm_experiment():
    files = [file for file in os.listdir(PATH)
             if ("tag1" in file or "tag2" in file)]
    X, y = read_dataset(PATH, files, segments_size=256, format_segments=segments_2d)

    start = time()
    categorize_chips.chip_type_svm(X, y)
    print("\nExecution time: %s [s]" % (time() - start))


def chip_type_cnn():
    # TODO Balance the amount of data between chip types?
    files = [file for file in os.listdir(PATH) if ".nfc" in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    for i, v in enumerate(y):
        v = int(v)
        if v in NTAG213:
            y[i] = 0
        elif v in MIFARE:
            y[i] = 1
        elif v in FELICA:
            y[i] = 2

    build_cnn(X, y, epochs=10)


def identify_tag():
    files = [file for file in os.listdir(PATH) if ".nfc" in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    build_cnn(X, y, 10)


def build_cnn(X, y, epochs):
    # Split data into train, validation and test data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, 0.7, 0.2, 0.1)

    # Build model and output its structure
    shape = (None,) + X_train.shape[1:]
    model = rfmlcnn.RFMLCNN(nb_outputs=len(set(y)), input_shape=shape)
    model.summary()

    # Configure model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model and adjust with validation set
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    print(history.history)
    # TODO Cross validation?

    # Evaluate model with test set
    # TODO put that in an evaluate module
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # TODO model.save()


if __name__ == '__main__':
    chip_type_cnn()

    identify_tag()