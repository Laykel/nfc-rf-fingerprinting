from pathlib import Path

import numpy as np

from dataset.configuration import load_conf, save_conf
from dataset.format import read_dataset
from learn.build import reload_model, build_cnn, build_svm
from learn.evaluate import analyse_model

CONF = Path("experiment.json")
MODELS = Path("../saved_models")


def train_model():
    print("Load experiment configuration_________")
    conf = load_conf(CONF)
    print("Read dataset__________________________")
    X, y = read_dataset(conf)

    print("Build model and train it______________")
    if conf['model']['type'] == "svm":
        build_svm(X, y)
    else:
        save_path = build_cnn(X, y, conf['model'])
        print("Save model and performance data_______")
        save_conf(CONF, save_path)


def load_model(model_path):
    print("Load experiment configuration_________")
    conf = load_conf(model_path / CONF)
    print("Read dataset__________________________")
    X, y = read_dataset(conf)
    X = np.expand_dims(X, axis=3)

    print("Reload model__________________________")
    model = reload_model(model_path / "model.tf", conf, X.shape)
    print("Analyse model performance_____________")
    analyse_model(model, X, y, conf['data']['classes'], model_path / "rerun-model")


if __name__ == '__main__':
    train_model()
    # load_model(MODELS / "youssef-ds2-all-640-fixed")
