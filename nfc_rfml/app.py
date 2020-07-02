#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from learn import categorize_chips

"""
This module is the entry point for the nfc-rfml project.
It contains the definition of each experiment performed for NFC Radio Frequency Machine Learning purposes.
"""

PATH = "../data/dataset/1"


def run():
    categorize_chips.chip_type_svm(PATH)


if __name__ == '__main__':
    run()
