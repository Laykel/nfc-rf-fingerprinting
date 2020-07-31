#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from utils.timer import timer
from preprocess.format import read_dataset, labels_as_chip_type, windows_2d
from learn.build import build_svm, build_cnn

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

DS1 = Path("../data/dataset/1")
DS2 = Path("../data/dataset/2")


def identify_tag():
    # All tags but the FeliCa one (9)
    tags = [1, 2, 3, 4, 5, 6, 7, 8]
    X, y = read_dataset(DS2, tags, window_size=512)
    build_cnn(X, y, epochs=200, early_stopping=False)


def among_ntag():
    # All NTAG213 tags
    tags = [1, 2, 3, 4, 5]
    X, y = read_dataset(DS2, tags, window_size=256)
    build_cnn(X, y, epochs=150, early_stopping=False)


def chip_type_cnn():
    # One tag of each type
    tags = [1, 6, 9]
    X, y = read_dataset(DS1, tags, window_size=512, filter_peaks=False)
    print(X.shape)

    labels_as_chip_type(y)

    build_cnn(X, y, epochs=50)


@timer
def svm_experiment():
    tags = [1, 2, 3, 4, 5, 6, 7, 8]
    X, y = read_dataset(DS2, tags, window_size=256, format_windows=windows_2d)

    # labels_as_chip_type(y)

    build_svm(X, y)


if __name__ == '__main__':
    chip_type_cnn()
