#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from preprocess.format import read_dataset, segments_2d, split_data

"""
...
"""


def chip_type_svm(path):
    # Get only the first recording for each tag
    files = [file for file in os.listdir(path)
             if ("tag1" in file or "tag2" in file)]
    print(files)
    X, y = read_dataset(path, files, segments_size=256, format_segments=segments_2d)

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = SVC()
    model.fit(X_train, y_train)

    training_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("-------------------------------------------")
    print("Training performance")
    print(classification_report(y_train, training_pred))
    print("-------------------------------------------")
    print("Testing performance")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-------------------------------------------")


if __name__ == "__main__":
    PATH = "../../data/dataset/1"
    chip_type_svm(PATH)
