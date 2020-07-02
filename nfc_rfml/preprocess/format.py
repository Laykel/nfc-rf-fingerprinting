#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy

"""
This module provides functions to load I/Q signals datasets in memory, formatting them as necessary for learning.
"""

# TODO Function to return metadata
# The IDs of tags of a given model
NTAG213 = (1, 2, 3, 4, 5)
MIFARE = (6, 7, 8)
FELICA = (9,)


def partition(lst, n):
    """Generate as many n-sized segments as possible from lst (the last segment may be smaller)
    :param lst: The list to partition
    :param n: The partition size
    :return: A list of partitions of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def segments_2d(segments):
    """Store the segments in simple arrays with the real parts first and then the imaginary parts.
    :param segments: A list of data segments with complex values
    :return: A list of arrays with all the real parts followed by the imaginary parts
    """
    return [np.append(np.real(lst), [np.imag(lst)]) for lst in list(segments)]


def segments_3d(segments):
    """Store the segments in 2d arrays with one dimension for the real parts and one for the imaginary parts.
    :param segments: A list of data segments with complex values
    :return: A list of two-dimensional arrays with each a list for real parts and a list for imaginary parts
    """
    return [np.vstack((np.real(lst), np.imag(lst))) for lst in list(segments)]


def read_dataset(path, files, segments_size=256, format_segments=segments_3d):
    """
    Sort and read the given files as complex numbers and partition them in smaller segments.
    Then, format these segments using a a given function and store them in a list of training/testing data.
    Finally, build a list of labels using the filename.

    :param path: The path to the folder where the data files are stored
    :param files: The file names of the files to be considered
    :param segments_size: The wanted size for the data segments
    :param format_segments: The function with which to format the signal segments
    :return: A couple of numpy arrays in the form (formatted data, labels)
    """
    files.sort()

    X = []  # The training/testing data
    labels = []  # The associated labels

    for file in files:
        # Read each capture
        signal = np.fromfile(os.path.join(path, file), dtype=scipy.complex64)

        # Partition the signals in segments of 256 samples
        segments = list(partition(signal, segments_size))
        # Format segments and add them to the collection of training/testing data
        X.extend(format_segments(segments))

        labels.extend([file[3]] * len(segments))  # TODO Calculate value of label through function

    return np.array(X), np.array(labels)


def read_metadata(path):
    # TODO
    pass


def _test():
    PATH = "../../data/dataset/1"
    # Get only the first recording for each tag
    files = [file for file in os.listdir(PATH) if file.endswith(".nfc") and "-1" in file]

    X, y = read_dataset(PATH, files)
    print(X.shape, y.shape)

    X, y = read_dataset(PATH, files, segments_size=512, format_segments=segments_2d)
    print(X.shape, y.shape)


if __name__ == "__main__":
    _test()
