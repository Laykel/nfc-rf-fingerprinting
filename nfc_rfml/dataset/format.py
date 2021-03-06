import os
import numpy as np
from scipy import complex64
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

"""
This module provides functions to load I/Q signals datasets in memory, formatting them as necessary for learning.
"""


def partition(lst, n):
    """Generate as many n-sized segments as possible from lst (the last segment may be smaller).
    :param lst: The list to partition
    :param n: The partition size
    :return: A generator of partitions of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def windows_2d(windows):
    """Store the segments in simple arrays with the real parts first and then the imaginary parts.
    :param windows: A list of partitions each containing a fixed number of complex numbers
    :return: A list of arrays with all the real parts followed by all the imaginary parts
    """
    return [np.append(np.real(window), [np.imag(window)]) for window in windows]


def windows_3d(windows):
    """Store the segments in 2d arrays with one dimension for the real parts and one for the imaginary parts.
    :param windows: A list of partitions each containing a fixed number of complex numbers
    :return: A list of two-dimensional arrays with each an array for real parts and one for imaginary parts
    """
    return np.stack((np.real(windows), np.imag(windows)), axis=1)


def filter_peaks_windows(signal, window_size, format_windows, height=0.1, threshold=0.005):
    """
    Detect a non-trivial part of the signal and create windows of `segments_size` samples, similarly as described by
    Youssef et al. in their paper (see bibliography).

    :param signal: The I/Q signal as a 1D array of complex numbers
    :param window_size: The wanted size for the signal windows (data segments)
    :param format_windows: The function with which to format the signal into windows
    :param height: The minimum height for a peak to be considered
    :param threshold: The prominence parameter used to find peaks in the signal
    :return: A list of windows filtered using a peak finding method and formatted according to given function
    """
    mags = np.abs(signal)
    # Detect peaks higher than height and with vertical distance to neighbours higher than threshold
    indices, _ = find_peaks(mags, height=height, threshold=threshold)
    windows = []

    # Partition the signal into windows
    while indices.size != 0:
        start = indices[0]
        indices = indices[indices > start + window_size]
        windows.append(signal[start:start + window_size])

    # Remove the last window, if it is truncated
    if len(windows[-1]) < window_size:
        windows = windows[:-1]

    return format_windows(windows)


def tags_files(path, tags):
    """Inspect the files in the given path and return the ones which contain the tag numbers specified.
    :param path: The path to the dataset's folder
    :param tags: The tag numbers whose signals have to be read
    :return: The file names which contain the necessary numbers
    """
    tag_names = [f"tag{x}" for x in tags]

    files = [file for file in os.listdir(path) if ".nfc" in file
             if any((True for name in tag_names if name in file))]
    files.sort()

    n = sum(1 for file in files if tag_names[0] in file)
    file_groups = list(zip(*[iter(files)] * n))

    return file_groups


def load_data(path, file_groups):
    """Read data from the raw files and concatenate signals from the same tags in a dataset array
    :param path: The path to the data files
    :param file_groups: The way in which to concatenate the files
    :return: An array of the appended signals in order of label
    """
    data = []

    for files in file_groups:
        # Read each capture
        signal = np.array([])
        for file in files:
            signal = np.append(signal, np.fromfile(os.path.join(path, file), dtype=complex64), 0)

        data.append(signal)

    return data


def harmonize_length(data, labels):
    """Harmonize the length of the different labels, make sure each one has the same amount of data
    :param data: The dataset containing all data
    :param labels: The labels for the data
    :return: The harmonized dataset and the labels correctly set for this dataset
    """
    # Find the smallest number of windows in a class
    min_window_number = min([len(x) for x in data])

    X, y = [], []
    # Go through the added data and truncate it to ensure a balanced dataset
    for label in labels:
        X.extend(data[label][:min_window_number])
        y.extend([label] * min_window_number)

    X, y = np.array(X), np.array(y)

    return X, y


def read_dataset(data_conf):
    """
    Sort and read the given files as complex numbers and partition them in smaller segments.
    Then, format these segments using a a given function and store them in a list of training/testing data.
    Finally, build a list of labels using the filename.

    :param data_conf: A dictionary containing the experiment's parameters
    :return: A couple of numpy arrays in the form (formatted data, labels)
    """
    files_per_tag = tags_files(data_conf['datapath'], data_conf['tags'])
    tags = load_data(data_conf['datapath'], files_per_tag)

    data = []
    for tag in tags:
        # Format signal in segments and add them to the collection of training/testing data
        if data_conf['filter']:
            formatted = filter_peaks_windows(tag, data_conf['windowsize'], data_conf['windows'])
        else:
            windows = list(partition(tag, data_conf['windowsize']))
            formatted = data_conf['windows'](windows)

        data.append(formatted)

    X, y = harmonize_length(data, data_conf['classes'])

    # Normalize our data
    if data_conf['normalize']:
        max_value = np.abs(X).max()
        X = X / max_value

    return X, y


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
    nb_classes = len(set(y))
    y = to_categorical(y.astype(int), nb_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    shuffle=False)

    dimensions = len(X.shape)
    X_train = np.expand_dims(X_train, axis=dimensions)
    X_val = np.expand_dims(X_val, axis=dimensions)
    X_test = np.expand_dims(X_test, axis=dimensions)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
