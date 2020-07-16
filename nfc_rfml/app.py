#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from time import time
from preprocess.format import read_dataset, split_data
from learn import categorize_chips
from learn.models import rfmlcnn

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

PATH = "../data/dataset/1"


def chip_type1():
    start = time()
    categorize_chips.chip_type_svm(PATH)
    print("\nExecution time: %s [s]" % (time() - start))


def cnn_test():
    files = [file for file in os.listdir(PATH) if ".nfc" in file]
    print(files)
    X, y = read_dataset(PATH, files, segments_size=512)

    # Split data into train, validation and test data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, 0.7, 0.2, 0.1)

    # Build model and output its structure
    shape = (None,) + X_train.shape[1:]
    model = rfmlcnn.RFMLCNN(nb_outputs=9, input_shape=shape)
    model.summary()

    # Configure model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model and adjust with validation set
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3)
    print(history.history)

    # Evaluate model with test set
    # TODO put that in an evaluate module
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # TODO always save model


def main():
    # chip_type1()
    cnn_test()


if __name__ == '__main__':
    main()
