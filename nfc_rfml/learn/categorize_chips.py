#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from nfc_rfml.preprocess.format import read_dataset, segments_2d

"""
...
"""


def chip_type_svm(path):
    # Get only the first recording for each tag
    files = [file for file in os.listdir(path)
             if ("tag1" in file or "tag6" in file)
             and "-1" in file or "-2" in file]
    X, y = read_dataset(path, files, format_segments=segments_2d)

    # TODO Think about validation data
    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    PATH = "../../data/dataset/1"
    chip_type_svm(PATH)
