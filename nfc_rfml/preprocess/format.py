import os
import numpy as np
import scipy.signal as sig
from scipy import complex64
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

"""
This module provides functions to load I/Q signals datasets in memory, formatting them as necessary for learning.
"""

# TODO Function to return metadata
# The IDs of tags of a given model
NTAG213 = (1, 2, 3, 4, 5)
MIFARE = (6, 7, 8)
FELICA = (9,)


def labels_as_chip_type(y):
    """
    TODO
    :param y:
    :return:
    """
    for i, v in enumerate(y):
        if v in NTAG213:
            y[i] = 0
        elif v in MIFARE:
            y[i] = 1
        elif v in FELICA:
            y[i] = 2


def partition(lst, n):
    """Generate as many n-sized segments as possible from lst (the last segment may be smaller).
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


def segments_peaks(segments):
    # TODO Clean up
    kept = segments
    return []


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
        signal = np.fromfile(os.path.join(path, file), dtype=complex64)

        # Partition the signals in segments of 256 samples
        segments = list(partition(signal, segments_size))
        # Format segments and add them to the collection of training/testing data
        X.extend(format_segments(segments))

        labels.extend([int(file[3])] * len(segments))  # TODO Calculate value of label through function

    return np.array(X), np.array(labels)


def split_data(X, y, train_ratio, validation_ratio, test_ratio):
    """
    Split the data and the labels in given portions of training, validation and testing data. Also converts the labels
    to categorical data.

    :param X: Data to split in portions defined in the ratio parameters
    :param y: Labels to split in the same manner as the data
    :param train_ratio: The wanted percentage of training data
    :param validation_ratio: The wanted percentage of validation data
    :param test_ratio: The wanted percentage of test data
    :return: Three couples with training data and labels, validation data and labels, and test data and labels
    """
    # https://datascience.stackexchange.com/a/53161
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    shuffle=False)

    # TODO determine axis with input dimensionality
    X_train = np.expand_dims(X_train, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    nb_classes = len(set(y))

    y_train = to_categorical(y_train.astype(int) - 1, nb_classes)
    y_val = to_categorical(y_val.astype(int) - 1, nb_classes)
    y_test = to_categorical(y_test.astype(int) - 1, nb_classes)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _test():
    PATH = "../../data/dataset/1"
    # Get only the first recording for each tag
    files = [file for file in os.listdir(PATH) if file.endswith(".nfc") and "-1" in file]

    X, y = read_dataset(PATH, files, segments_size=512, format_segments=segments_2d)
    print(X.shape, y.shape)

    X, y = read_dataset(PATH, files)
    print(X.shape, y.shape)

    train, validate, test = split_data(X, y, 0.7, 0.2, 0.1)
    print(train[0].shape, validate[0].shape, test[0].shape)

    print(y)
    labels_as_chip_type(y)
    print(y)


if __name__ == "__main__":
    _test()
