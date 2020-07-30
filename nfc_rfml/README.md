# Radio frequency machine learning for NFC devices

## Description

This project is a collection of experiments and utilities on the subject of Radio Frequency Machine Learning (RFML).
Its goal is to be able to discriminate between NFC tags using raw wave data (I/Q signals).

The data it uses is in the data folder of this repository. For more information on the project, the acquisition setup
or any conceptual topic, see the project's [report](../report/final-report.pdf).

## Project structure

- `learn`: Contains the deep learning models used and utilities to build and train them, and to evaluate their performance.
    - `models`: This is where the neural networks are defined.
- `preprocess`: The modules that manipulate the datasets live in this package.
- `utils`: Some utility methods and decorators, e.g. a `timer` decorator.

## How to run the project

To run the main experiment, you only need to do the following.

```
# Install dependencies (after creating a virtual env as you usually do)
pip install -r requirements.txt

python app.py
```

This will read and process the data, train the model and output the performance, some stats and useful plots to the
`saved_model` folder, one level up from the `nfc_rfml` directory.

As this is more an exploration of the possibilities in the field of RFML, there is no fancy interface. If you want
to change the type of experiment or the processing done to the data, you'll have to manipulate the `app.py` file.