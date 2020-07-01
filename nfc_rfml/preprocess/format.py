#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

"""
This module provides functions to load I/Q signals datasets in memory, formatting them as necessary for learning.
"""

PATH = "../../data/dataset/1"

# TODO Function to return metadata
# The IDs of tags of a given model
NTAG213 = (1, 2, 3, 4, 5)
MIFARE = (6, 7, 8)
FELICA = (9,)


def partition(lst, n):
    """Generate as many n-sized segments as possible from lst
    :param lst: The list to partition
    :param n: The partition size
    :return: A list of partitions of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_dataset(path, files, segments_size=256):
    """
    Sort and read the given files as complex numbers and partition them in smaller segments.
    Then, separate the real and imaginary parts of these segments' elements in two float-valued arrays.
    Finally, store each of those two-dimensional arrays in an array and build a labels list using the filename.

    :param path: The path to the folder where the data files are stored
    :param files: The file names of the files to be considered
    :param segments_size: The wanted size for the data segments
    :return: A couple of numpy arrays in the form (formatted data, labels)
    """
    files.sort()

    training = []
    labels = []

    for file in files:
        # Read each capture
        signal = np.fromfile(os.path.join(path, file), dtype=scipy.complex64)

        # Partition the signals in segments of 256 samples
        segments = partition(signal, segments_size)
        # Store these segments in a two-dimensional array
        # (one dimension for the real parts and one for the imaginary parts)
        training.extend([np.vstack((np.real(lst), np.imag(lst))) for lst in list(segments)])

        labels.extend([file[3]] * 9000)  # TODO Calculate value of label through function

    return np.array(training), np.array(labels)


def main():
    # Get only the first recording for each tag
    files = [file for file in os.listdir(PATH) if file.endswith(".nfc") and "-1" in file]
    X, y = read_dataset(PATH, files)

    print(X.shape, y.shape)

    # TODO Maybe don't train_test_split in this module
    # TODO Think about validation data
    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    main()
