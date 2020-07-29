# Datasets description

We have produced a number of data collections comprising recordings of NFC transmissions between an active reader (smartphone) and a number of passive tags.

The transmissions are stored in binary format (raw bytes) with each 4 bytes representing a (32b) floating-point number. As the samples are represented as complex numbers (see report), the numbers are interleaved: the first float is the real part of the sample and the second is the imaginary part. They can easily be read as a list of complex numbers in python (with numpy) using the following line.

```py
import numpy as np
signal = np.fromfile("path/to/file.nfc", dtype=scipy.complex64)
```

Each of the subfolders contains a set of recordings, which we describe below.

## NFC tags inventory

| Name | NFC Type | Standard     | Chip              | Writable | Storage | Bit rate           |
|:-----|:---------|:-------------|:------------------|:---------|:--------|:-------------------|
| tag1 | NFC-A    | ISO 14443-3A | NTAG213           | Yes      | 137B    | 106kb/s            |
| tag2 | NFC-A    | ISO 14443-3A | NTAG213           | Yes      | 137B    | 106kb/s            |
| tag3 | NFC-A    | ISO 14443-3A | NTAG213           | Yes      | 137B    | 106kb/s            |
| tag4 | NFC-A    | ISO 14443-3A | NTAG213           | Yes      | 137B    | 106kb/s            |
| tag5 | NFC-A    | ISO 14443-3A | NTAG213           | Yes      | 137B    | 106kb/s            |
| tag6 | NFC-A    | ISO 14443-3A | Mifare Classic 1k | Yes      | 716B    | 106kb/s            |
| tag7 | NFC-A    | ISO 14443-3A | Mifare Classic 1k | Yes      | 716B    | 106kb/s            |
| tag8 | NFC-A    | ISO 14443-4  | Mifare Classic 4k | No       | ~4kB    | 106kb/s            |
| tag9 | FeliCa   | JIS 6319-4   | RC-S967           | No       | 208B    | 212kb/s or 424kb/s |

## General parameters for the acquisition

We use an Airspy HF+ SDR device with an NFC antenna for all our captures. In order to capture the NFC communication, we set the center frequency to 13.56 MHz.

For more details on the setup, please refer to the report.

## Dataset 1

This dataset was built while trying to introduce as little variability as possible. The positions of the tags and reader was kept as similar as possible between recordings. Each recording was made using `scripts/capture.py` with `--time 3`.

- Number of samples per seconds: 768'000
- 3 recordings of each of the 9 tags
- Length of a recording: 3 seconds
- Size of a recording on disk: 18.432 MB
- Total size of the dataset: 497.664 MB
- (Content of tags 1 through 7: 36B of the 'A' character.)

## Dataset 2

This dataset was built in the same manner as the previous one but the captures are much longer and more variable. We were also able to increase the sample rate thanks to a firmware update of the Airspy HF+. The goal here was to address the shortcomings of our first dataset in terms of volume and variability. To achieve that, we made the capture time longer and also captured the very start of the communications: the transient part.

There is no synchronization for the beginning of the communications, so the transient part can happen anywhere within the 2 first seconds of the recording. We chose to keep theses datasets as raw as possible.

Each recording was made using `scripts/capture.py` with `--samplerate 912000` and `--time 30`.

- Distance between reader and tags: ~1.8 cm

- Number of samples per seconds: 912'000
- 3 recordings of each of the first 8 tags
- Length of a recording: 20 seconds
- Size of a recording on disk: 145.92 MB
- Total size of the dataset: 3502.08 MB
- (Content of tags 1 through 7: 36B of the 'A' character.)

## Dataset 9

This dataset is numbered 9 and will probably not be used because the tags' responses are very weak. We keep it as it might still serve in a future experiment. Each recording was made using `scripts/capture.py` with `--time 5`.

- 2 recordings of each of the 9 tags
- Length of a recording: 5 seconds
- Number of samples per recording: 5 * 768'000 = 3'840'000 samples
- (Content of tags 1 through 7: 36B of the 'A' character.)
