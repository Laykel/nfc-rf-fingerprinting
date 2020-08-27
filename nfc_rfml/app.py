from pathlib import Path

import numpy as np

from dataset.configuration import load_conf, save_conf
from dataset.format import read_dataset
from learn.build import reload_model, build_cnn
from learn.evaluate import analyse_model

CONF = Path("experiment.json")
MODELS = Path("../saved_models")


def train_model():
    conf = load_conf(CONF)
    X, y = read_dataset(conf)

    save_path = build_cnn(X, y, conf['model']['epochs'], conf['model']['batchsize'], conf['model']['earlystopping'])

    save_conf(CONF, save_path)


def load_model(model_path):
    conf = load_conf(model_path / CONF)
    X, y = read_dataset(conf)
    X = np.expand_dims(X, axis=3)

    model = reload_model(model_path / "model.tf", conf, X.shape)

    analyse_model(model, X, y, conf['data']['classes'])


if __name__ == '__main__':
    train_model()
    # load_model(MODELS / "youssef-all-640-76%")
