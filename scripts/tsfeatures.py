import os
from pathlib import Path
import numpy as np
from scipy import complex64
from scipy.signal import find_peaks
import pandas as pd

from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute


def find_transfers(signal, window_size=256):
    mags = np.abs(signal)
    # Detect peaks higher than height and with vertical distance to neighbours higher than threshold
    indices, _ = find_peaks(mags, height=0.1, threshold=0.005)
    filtered = []

    while indices.size != 0:
        start = indices[0]
        indices = indices[indices > start + window_size]
        filtered.extend(signal[start:start + window_size])

    return filtered


def main():
    path = Path("../data/dataset/2")
    files = [file for file in os.listdir(path) if ".nfc" in file
             and "-1" in file]

    labels = {
        1: 0,
        6: 1,
        9: 2
    }
    tables = []

    for file in files:
        signal = np.fromfile(os.path.join(path, file), dtype=complex64)
        filtered = find_transfers(signal)[:20000]
        table = np.vstack((np.real(filtered),
                           np.imag(filtered),
                           # np.full(len(filtered), labels[int(file[3])]),
                           np.full(len(filtered), int(file[3]) - 1),
                           np.arange(len(filtered))))
        tables.append(table)

    df = pd.DataFrame(np.transpose(np.hstack(tables)), columns=["I", "Q", "class", "time"])
    df = df.astype({"class": int, "time": int})
    print(df)

    classes = [(x, x) for x in df["class"].to_numpy()]
    y = pd.Series(dict(classes))

    extracted_features = extract_features(df, column_id="class", column_sort="time",
                                          default_fc_parameters=EfficientFCParameters())
    # extracted_features = extract_features(df, column_id="class", column_sort="time")
    extracted_features.to_csv("features.csv")
    print(extracted_features)

    impute(extracted_features)
    filtered_features = select_features(extracted_features, y)
    filtered_features.to_csv("selected_features.csv")
    print(filtered_features)


if __name__ == "__main__":
    main()
