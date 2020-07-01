#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot as plt
import os

"""
This script contains experiments for finding interesting parts in our signals and generating visualisations
"""

PATH = "../../data/dataset/1"

SAMP_RATE = int(768e3)
NFFT = 1024


def signal_psd(data, samp_rate, nfft, output):
    plt.figure(figsize=(30, 10))

    plt.psd(data, NFFT=nfft, Fs=samp_rate)
    plt.title("PSD")

    plt.savefig("{}.png".format(output), bbox_inches='tight')
    plt.close()


def signal_ivsq(data, output):
    """Save a plot of the samples' I vs Q components in the `output` file"""
    plt.figure(figsize=(30, 10))

    plt.plot(np.real(data), 'r.', np.imag(data), 'b.')
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.savefig("{}.png".format(output), bbox_inches='tight')
    plt.close()


def signal_magnitude(data, peak, output):
    """Save a plot of the samples' magnitudes in the `output` file"""
    plt.figure(figsize=(30, 10))

    plt.plot(np.abs(data), 'b.')

    plt.xticks(range(0, 4500, 500), range(peak - 2500, peak + 2001, 500))
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")

    plt.savefig("{}.png".format(output), bbox_inches='tight')
    plt.close()


def main():
    files = [file for file in os.listdir(PATH)]
    files.sort()

    print("Files to generate visualizations for:")
    print(files)

    for file in files:
        data = np.fromfile(os.path.join(PATH, file), dtype=scipy.complex64)
        mags = np.abs(data)

        # Find peak in signal
        peak = signal.find_peaks(mags[2000:], 0.277)[0][0] + 2000
        start, end = peak - 2000, peak + 2000
        print(file[:-4], "start:", start, "end:", end)

        # signal_ivsq(data[start:end], "figs/IQ/{}".format(file[:-4]))
        signal_magnitude(data[start:end], peak, "figs/magnitudes/{}".format(file[:-4]))
        # signal_psd(data[start:end], SAMP_RATE, NFFT, "figs/PSD/{}".format(file[:-4]))


if __name__ == "__main__":
    main()
