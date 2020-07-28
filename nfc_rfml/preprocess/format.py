import os
import numpy as np
from scipy import complex64
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

"""
This module provides functions to load I/Q signals datasets in memory, formatting them as necessary for learning.
"""


def labels_as_chip_type(y):
    """Transform the individual tag labels given as parameter into chip type labels
    :param y: The labels of individual tags to transform to chip type labels
    """
    NTAG213 = (0, 1, 2, 3, 4)
    MIFARE = (5, 6, 7)
    FELICA = (8,)

    y[np.isin(y, NTAG213)] = 0
    y[np.isin(y, MIFARE)] = 1
    y[np.isin(y, FELICA)] = 2


def partition(lst, n):
    """Generate as many n-sized segments as possible from lst (the last segment may be smaller).
    :param lst: The list to partition
    :param n: The partition size
    :return: A generator of partitions of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def normalize_amplitude(signal):
    """Normalize both components of the signal in the range [-1,1]
    :param signal: The I/Q signal to be normalized
    :return: The I/Q signal in the same format as before, but with both components normalized in the range [-1,1]
    """
    real = np.real(signal)
    imag = np.imag(signal)
    max_val = abs(max(max(real.max(), imag.max()),
                      min(real.min(), imag.min()),
                      key=abs))

    # Convert the normalized lists to complex
    return (real / max_val) + 1j * (imag / max_val)


def windows_2d(windows):
    """Store the segments in simple arrays with the real parts first and then the imaginary parts.
    :param windows:
    :return: A list of arrays with all the real parts followed by all the imaginary parts
    """
    return [np.append(np.real(window), [np.imag(window)]) for window in windows]


def windows_3d(windows):
    """Store the segments in 2d arrays with one dimension for the real parts and one for the imaginary parts.
    :param windows:
    :return: A list of two-dimensional arrays with each an array for real parts and one for imaginary parts
    """
    return list(zip(np.real(windows), np.imag(windows)))


def cut_windows(signal, windows_size, windows_format):
    windows = list(partition(signal, windows_size))
    return windows_format(windows)


def filter_peaks_windows(signal, windows_size, windows_format):
    """
    Detect a non-trivial part of the signal and create windows of `segments_size` samples, similarly as described by
    Youssef et al. in their paper (see bibliography).

    :param signal: The I/Q signal as a 1D array of complex numbers
    :param windows_size: The wanted size for the signal windows (data segments)
    :param windows_format: The function to use to format the
    :return: ...
    """
    mags = np.abs(signal)
    # TODO calculate the value instead of hardcoding 0.2
    indices = np.where(mags < 0.2)[0]
    windows = []

    while indices.size != 0:
        start = indices[0]
        indices = indices[windows_size:]
        windows.append(signal[start:start + windows_size])

    # Remove the last one, since it can be truncated
    return windows_format(windows[:-1])


def tags_files(path, tags):
    """
    TODO
    :param path:
    :param tags:
    :return:
    """
    tags_names = [f"tag{x}" for x in tags]

    files = [file for file in os.listdir(path) if ".nfc" in file
             if any((True for x in tags_names if x in file))]
    files.sort()

    return files


def read_dataset(path, tags, windows_size=256, windows_format=windows_3d, filter_peaks=False, normalize=False):
    """
    Sort and read the given files as complex numbers and partition them in smaller segments.
    Then, format these segments using a a given function and store them in a list of training/testing data.
    Finally, build a list of labels using the filename.

    :param path: The path to the folder where the data files are stored
    :param tags: A list of tags numbers as defined in the dataset's description
    :param windows_size: The wanted size for the signal windows (data segments)
    :param windows_format: The function with which to format the signal into windows
    :param normalize: Whether to normalize the signal in terms of amplitude
    :return: A couple of numpy arrays in the form (formatted data, labels)
    """
    files = tags_files(path, tags)

    X = []  # The training/testing data
    labels = []  # The associated labels

    for file in files:
        # Read each capture
        signal = np.fromfile(os.path.join(path, file), dtype=complex64)
        if normalize:
            signal = normalize_amplitude(signal)

        # Format signal in segments and add them to the collection of training/testing data
        if filter_peaks:
            formatted = filter_peaks_windows(signal, windows_size, windows_format)
        else:
            formatted = cut_windows(signal, windows_size, windows_format)

        X.extend(formatted)

        # Get the label from the filename
        labels.extend([int(file[3]) - 1] * len(formatted))

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
    PATH = "../../data/dataset/2"
    tags = range(6)

    X, y = read_dataset(PATH, tags, windows_format=windows_2d)
    print(X.shape, y.shape)
    del X, y

    X, y = read_dataset(PATH, tags)
    print(X.shape, y.shape)
    del X, y

    X, y = read_dataset(PATH, tags, filter_peaks=True)
    print(X.shape, y.shape)

    train, validate, test = split_data(X, y, 0.7, 0.2, 0.1)
    print(train[0].shape, validate[0].shape, test[0].shape)

    print(y)
    labels_as_chip_type(y)
    print(y)


if __name__ == "__main__":
    _test()
