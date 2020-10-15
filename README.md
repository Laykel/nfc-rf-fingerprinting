# RF fingerprinting for NFC device identification

## Description

(See the project's [report](report/final-report.pdf) for a precise account of every step.)

### Abstract

**RF fingerprinting** is a technique that allows the identification of radio transmitters by extracting small imperfections in their spectrum. These imperfections are caused by tiny manufacturing differences in the devices' analog components. Using **Software-Defined Radio (SDR)** equipment, we can analyse this spectrum in order to extract the aforementioned differences and identify a device.

Such techniques can be used on any type of radio transmission: Bluetooth, BLE, WiFi, LTE, etc. This project aims to use RF fingerprinting on NFC devices. Indeed, NFC is often used in access control and payment applications but many implementations are vulnerable to **relay attacks**. Spoofing the imperfections in an emitter's radio spectrum is close to impossible at the present time, since it is essentially a hardware signature. This is why a technique like the one described here would be a valuable additional security layer.

The goal of this project is to determine if RF fingerprinting of **NFC devices** could be used as an authentication technique, in order to prevent relay attacks.

### State of the art

A popular way of extracting the fine features generated by the hardware imperfections is through Convolutional Neural Networks (CNNs). This class of deep neural networks has had a lot of success in image and video recognition, recommendation systems, image classification and natural language processing. It is not surprising, therefore, that researchers have shown promising results in RF fingerprinting applications.

Most of the research work is focused on the problem of identifying each transmitter specifically. Their approaches show good results but as soon as the number of devices goes up, or as the environment gets more noisy, performance dwindles. Most solutions to this problem are not yet scalable, and not viable for real world applications.

This project differs in the type of devices considered and in the formulation of the main objective. Indeed, as a first step at least, our goal is to determine whether we can effectively differentiate a specific NFC device from all others.

### Context

This project is conducted in the context of my bachelor thesis at [HEIG-VD](https://heig-vd.ch/en).

- Department: Information and communication technologies
- Faculty: Information Technology and communication systems
- Orientation: Software engineering

## Structure

This repository's folders can be described as follows:

- `data` contains decoding tests as well as the `dataset` folder, ~~which contains the data captured for the project (stored using [Git LFS](https://git-lfs.github.com/)). See [the dataset readme](data/dataset/README.md).~~ (The datasets were removed from git LFS after the end of the project. Please open an issue if you're unable to create your own datasets.)
- `gnuradio` contains the different GRC flowgraphs used in the project.
- `nfc_rfml` contains the source code used to pre-process the signals, train the model, and test its performance. See [the project's readme](nfc_rfml/README.md).
- `notebooks` contains the notebooks used during the analysis and prototyping phase of the project.
- `report` contains the LaTeX sources and the compiled pdf of the report. It also contains the bibliography and figures used in it.
- `scripts` contains small programs like the acquisition script.

## How to run

The first step to run anything from this project is to clone the repo. (Warning, the datasets were removed from git LFS. Open an issue if you're unable to build your own datasets.) For more information on the main python project, see [the its readme](nfc_rfml/README.md).

### The acquisition script

This program is a very simple script based on a GNU Radio generated script. You do need GNU Radio installed (version 3.7 or higher) and a functioning python setup.

More importantly, the program uses the `osmocom source` block to try to connect to an Airspy HF+ device. You will probably need to adapt the source block to your hardware (see what GNU Radio Companion generates).

```
cd scripts
python capture.py your_filename.bin --time 4 --samplerate 500000 --freq 13560000
```

### The notebooks

```
# Install dependencies (after creating a virtual env as you usually do)
cd notebooks
pip install -r requirements.txt

# Run jupyter notebook or jupyter lab
jupyter lab
```
