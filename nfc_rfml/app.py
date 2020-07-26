#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from utils.timer import timer
from preprocess.format import read_dataset, labels_as_chip_type, segments_2d, segments_peaks
from learn.build import build_svm, build_cnn

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

PATH = Path("../data/dataset/1")


def identify_tag():
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag9" not in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    build_cnn(X, y, epochs=100)


def among_ntag():
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag1" in file or "tag2" in file or "tag3" in file
             or "tag4" in file or "tag5" in file]
    X, y = read_dataset(PATH, files, segments_size=256, format_segments=segments_peaks, normalize=True)

    build_cnn(X, y, epochs=100)


def chip_type_cnn():
    # One tag of each type
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag1" in file or "tag6" in file or "tag9" in file]
    X, y = read_dataset(PATH, files, segments_size=256)
    print(X.shape, y.shape)

    labels_as_chip_type(y)

    # Best params seem to be 256 points per segment, 500 samples per batch
    build_cnn(X, y, epochs=10)


@timer
def svm_experiment():
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag1" in file or "tag6" in file or "tag9" in file]
    X, y = read_dataset(PATH, files, segments_size=512, format_segments=segments_2d)

    labels_as_chip_type(y)

    build_svm(X, y)


if __name__ == '__main__':
    among_ntag()
