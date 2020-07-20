#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import time
from preprocess.format import read_dataset, segments_2d
from learn.build import build_svm, build_cnn

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

PATH = "../data/dataset/1"

# TODO Function to return metadata
# The IDs of tags of a given model
NTAG213 = (1, 2, 3, 4, 5)
MIFARE = (6, 7, 8)
FELICA = (9,)


def svm_experiment():
    files = [file for file in os.listdir(PATH)
             if ("tag1" in file or "tag2" in file)]
    X, y = read_dataset(PATH, files, segments_size=256, format_segments=segments_2d)

    start = time()
    build_svm(X, y)
    print("\nExecution time: %s [s]" % (time() - start))


def chip_type_cnn():
    # TODO Balance the amount of data between chip types?
    files = [file for file in os.listdir(PATH) if ".nfc" in file
             if "tag9" in file or "tag1" in file or "tag6" in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    for i, v in enumerate(y):
        if v in NTAG213:
            y[i] = 0
        elif v in MIFARE:
            y[i] = 1
        elif v in FELICA:
            y[i] = 2

    build_cnn(X, y, epochs=20)


def identify_tag():
    files = [file for file in os.listdir(PATH) if ".nfc" in file]
    X, y = read_dataset(PATH, files, segments_size=512)

    build_cnn(X, y, epochs=30)


if __name__ == '__main__':
    chip_type_cnn()
