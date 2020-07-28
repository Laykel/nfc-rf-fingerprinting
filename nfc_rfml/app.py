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

PATH = Path("../data/dataset/2")


def identify_tag():
    # All tags but the FeliCa one (9)
    tags = [1, 2, 3, 4, 5, 6, 7, 8]
    X, y = read_dataset(PATH, tags, windows_size=512)
    build_cnn(X, y, epochs=100)


def among_ntag():
    # All NTAG213 tags
    tags = [1, 2, 3, 4, 5]
    X, y = read_dataset(PATH, tags, windows_size=256)
    build_cnn(X, y, epochs=10)


def chip_type_cnn():
    # One tag of each type
    tags = [1, 6, 9]
    X, y = read_dataset(PATH, tags, windows_size=256)

    labels_as_chip_type(y)

    # Best params for this experiment seem to be 256 points per segment, 500 samples per batch
    build_cnn(X, y, epochs=150)


@timer
def svm_experiment():
    tags = [1, 6, 9]
    X, y = read_dataset(PATH, tags, windows_size=256, windows_format=windows_2d)

    labels_as_chip_type(y)

    build_svm(X, y)


if __name__ == '__main__':
    among_ntag()
