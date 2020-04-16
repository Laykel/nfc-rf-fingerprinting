# RF fingerprinting for NFC device identification

## Description

See the project's [report](report/bt-report.pdf) for a precise account of every step (includes the specification).

### Abstract

**RF fingerprinting** is a technique that allows the identification of radio transmitters by extracting small imperfections in their spectrum. These imperfections are caused by tiny manufacturing differences in the devices' analog components. Using **Software-Defined Radio (SDR)** equipment, we can analyse this spectrum in order to extract the aforementioned differences and identify a device.

Such technique can be used on any type of radio transmission: Bluetooth, BLE, WiFi, LTE, etc. This project aims to use RF fingerprinting on NFC devices. Indeed, NFC is often used in access control and payment applications but many implementations are vulnerable to **relay attacks**. Spoofing the imperfections in an emitter's radio spectrum is close to impossible at the present time, since it is essentially a hardware signature. This is why a technique like the one described here would be a valuable additional security layer.

The goal of this project is to determine if RF fingerprinting of **NFC devices** could be used as an authentication technique, in order to prevent relay attacks.

### State of the art

A popular way of extracting the fine features generated by the hardware imperfections is through Convolutional Neural Networks (CNNs). This class of deep neural networks has had a lot of success in image and video recognition, recommendation systems, image classification and natural language processing. It is not surprising, therefore, that researchers have shown promising results in RF fingerprinting applications.

Most of the research work is focused on the problem of identifying each transmitter specifically. Their approaches show good results but as soon as the number of devices goes up, or as the environment gets more noisy, performance dwindles. Most solutions to this problem are not yet scalable, and not viable for real world applications.

This project differs in the type of devices considered and in the formulation of the main objective. Indeed, as a first step at least, our goal is to determine whether we can effectively differentiate a specific NFC device from all others.

### Context

This project is conducted in the context of my bachelor thesis at [HEIG-VD](https://heig-vd.ch/en).

- Department: Information and communication technologies
- Sector: IT and communication systems
- Faculty: Software engineering

## Structure

This repository's folders can be described as follows:

- `data` contains the _gitignored_ recordings of NFC communications, as well as decoded transmissions in different formats.
- `gnuradio` contains the different GRC flowgraphs used in the project.
- `notebooks` contains the notebooks used during the testing and prototyping phase of the project.
- `report` contains the LaTeX sources and the compiled pdf of the report. It also contains the bibliography and figures used in it.
- `scripts` contains the final programs used to pre-process the signals, train the model, test it, and _deploy it?_ .

## How to run

```
# Get the data and unzip it
wget ...
# Clone the repo
git clone ...
# Install dependencies
pip install -r requirements.txt
# Run scripts / notebooks
...
```
