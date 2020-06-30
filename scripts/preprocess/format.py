#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

"""
This module contains
"""

PATH = "../../data/dataset/1"

NTAG213 = (1, 2, 3, 4, 5)
MIFARE = (6, 7, 8)
FELICA = (9,)


# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def partition(lst, n):  # TODO Adapt and change
    """Yield successive n-sized segments from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    # Get only the first recording for each tag
    files = [file for file in os.listdir(PATH) if file.endswith(".nfc") and "-1" in file]
    files.sort()

    training = []
    labels = []

    # Read each capture, partition the signals in segments of 256 samples and store these segments in a
    # two-dimensional array (one dimension for the real parts and one for the imaginary parts)
    for file in files:
        signal = np.fromfile(os.path.join(PATH, file), dtype=scipy.complex64)

        segments = partition(signal, 256)
        training.extend([np.vstack((np.real(lst), np.imag(lst))) for lst in list(segments)])

        labels.extend([file[3]] * 9000)  # TODO Calculate value of label through function

    training = np.array(training)
    labels = np.array(labels)
    print(training.shape, labels.shape)

    # Split data into train and test data
    # TODO Think about validation data
    X_train, X_test, y_train, y_test = train_test_split(training, labels, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    main()
