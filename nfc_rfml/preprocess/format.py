import os
import numpy as np
from detecta import detect_peaks
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
    indices = detect_peaks(mags, mph=height, threshold=threshold)
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


def read_dataset(path, tags, window_size=256, format_windows=windows_3d, filter_peaks=True, normalize=True):
    """
    Sort and read the given files as complex numbers and partition them in smaller segments.
    Then, format these segments using a a given function and store them in a list of training/testing data.
    Finally, build a list of labels using the filename.

    :param path: The path to the folder where the data files are stored
    :param tags: A list of tags numbers as defined in the dataset's description
    :param window_size: The wanted size for the signal windows (data segments)
    :param format_windows: The function with which to format the signal into windows
    :param filter_peaks: Whether to apply our peak detection function to the signal
    :param normalize: Whether to normalize the signal in terms of amplitude
    :return: A couple of numpy arrays in the form (formatted data, labels)
    """
    file_groups = tags_files(path, tags)

    data = []  # The training/testing data
    labels = []  # The associated labels

    for files in file_groups:
        # Read each capture
        signal = np.array([])
        for file in files:
            signal = np.append(signal, np.fromfile(os.path.join(path, file), dtype=complex64), 0)

        # Format signal in segments and add them to the collection of training/testing data
        if filter_peaks:
            formatted = filter_peaks_windows(signal, window_size, format_windows)
        else:
            windows = list(partition(signal, window_size))
            formatted = format_windows(windows)

        data.append(formatted)
        # Get the label from the filename
        labels.append(int(files[0][3]) - 1)

    # Find the class with the smallest number of windows
    min_window_number = min([len(x) for x in data])

    # Go through the added data and truncate it to ensure a balanced dataset
    X, y = [], []
    for idx, label in enumerate(labels):
        X.extend(data[idx][:min_window_number])
        y.extend([label] * min_window_number)

    X, y = np.array(X), np.array(y)
    if normalize:
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    shuffle=False)

    dimensions = len(X.shape)
    X_train = np.expand_dims(X_train, axis=dimensions)
    X_val = np.expand_dims(X_val, axis=dimensions)
    X_test = np.expand_dims(X_test, axis=dimensions)

    nb_classes = len(set(y))

    y_train = to_categorical(y_train.astype(int) - 1, nb_classes)
    y_val = to_categorical(y_val.astype(int) - 1, nb_classes)
    y_test = to_categorical(y_test.astype(int) - 1, nb_classes)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _test():
    PATH = "../../data/dataset/1"
    tags = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    X, y = read_dataset(PATH, tags, filter_peaks=True, normalize=True)
    print(X.shape, y.shape)

    train, validate, test = split_data(X, y, 0.7, 0.2, 0.1)
    print(train[0].shape, validate[0].shape, test[0].shape)


if __name__ == "__main__":
    _test()
