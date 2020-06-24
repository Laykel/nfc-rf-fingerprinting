import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot as plt
import os

PATH = "../../data/raw"

SAMP_RATE = int(768e3)
NFFT = 1024


def find_first_interesting_spot(data):
    """Return the index of the first significant part of the signal"""
    median_vector = signal.medfilt(data)
    print(signal.find_peaks(data, 0.6))
    return np.argmax(data > np.mean(median_vector))


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


def signal_magnitude(data, output):
    """Save a plot of the samples' magnitudes in the `output` file"""
    plt.figure(figsize=(30, 10))

    plt.plot(np.abs(data), 'b.')
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

        start = find_first_interesting_spot(np.abs(data))
        end = start + int(SAMP_RATE / 2)

        print(file[:-4], "start:", start, "end:", end)

        # signal_ivsq(data[start:end], "figs/IQ/{}".format(file[:-4]))
        signal_magnitude(data[start:end], "figs/magnitudes/{}".format(file[:-4]))
        # signal_psd(data[start:end], SAMP_RATE, NFFT, "figs/PSD/{}".format(file[:-4]))


if __name__ == "__main__":
    main()
