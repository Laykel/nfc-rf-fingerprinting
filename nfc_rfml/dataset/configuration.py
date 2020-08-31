import json
from shutil import copyfile

from dataset.format import windows_2d, windows_3d
from learn.models.youssef_cnn import YoussefCNN
from learn.models.riyaz_cnn import RiyazCNN


windows = {
    "2d": windows_2d,
    "3d": windows_3d
}

model_type = {
    "youssef": YoussefCNN,
    "riyaz": RiyazCNN
    # TODO SVM?
}


def load_conf(conf_path):
    with open(conf_path, "r") as file:
        conf = json.load(file)

    conf['data']['windows'] = windows[conf['data']['windows']]
    conf['model']['type'] = model_type[conf['model']['type']]
    return conf


def save_conf(conf_path, save_path):
    copyfile(conf_path, save_path / "experiment.json")
