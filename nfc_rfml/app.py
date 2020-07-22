#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import time
from preprocess.format import read_dataset, labels_as_chip_type, segments_2d
from learn.build import build_svm, build_cnn

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

PATH = "../data/dataset/1"


def svm_experiment():
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag9" in file or "tag1" in file or "tag6" in file]
    X, y = read_dataset(PATH, files, segments_size=512, format_segments=segments_2d)

    labels_as_chip_type(y)

    start = time()
    build_svm(X, y)
    print("\nExecution time: %s [s]" % (time() - start))


def chip_type_cnn():
    # TODO Balance the amount of data between chip types?
    # files = [file for file in os.listdir(PATH) if ".nfc" in file
    #          if "tag9" not in file and "tag4" not in file and "tag5" not in file]
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag9" in file or "tag1" in file or "tag6" in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    labels_as_chip_type(y)

    build_cnn(X, y, epochs=100)


def identify_tag():
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag9" not in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    build_cnn(X, y, epochs=100)


if __name__ == '__main__':
    chip_type_cnn()
