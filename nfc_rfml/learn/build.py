import os
from pathlib import Path
from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from dataset.format import split_data
from learn.models import youssef_cnn, riyaz_cnn
from learn.evaluate import evaluate_model

"""
Provide some do-all functions to configure, train and evaluate models.
"""


def reload_model(model_path, conf, Xshape):
    """
    TODO
    :param model_path:
    :param conf:
    :param Xshape:
    :return:
    """
    shape = (None,) + Xshape[1:]

    model = conf['model']['type'](nb_outputs=len(conf['data']['tags']), input_shape=shape)
    model.load_weights(str(model_path)).expect_partial()
    return model


def build_cnn(X, y, model_conf):
    # Split data into train, validation and test data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, 0.7, 0.2, 0.1)

    # Build model and output its structure
    shape = (None,) + X_train.shape[1:]
    model = model_conf['type'](nb_outputs=len(set(y)), input_shape=shape)

    # Configure model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Setup the folder structure
    dt = datetime.now().strftime("%Y-%m-%d %Hh%M")
    model_dir = Path(f"../saved_models/{dt}")
    os.makedirs(model_dir / "wrong-predictions")
    os.makedirs(model_dir / "correct-predictions")
    model_path = str(model_dir / "model.tf")

    # Save the best model to disk
    callbacks = [ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)]
    # Make sure the training stops when the performance stops getting better
    if model_conf['earlystopping']:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=8))

    # Train model and adjust with validation set
    history = model.fit(X_train, y_train,
                        epochs=model_conf['epochs'],
                        batch_size=model_conf['batchsize'],
                        callbacks=callbacks,
                        validation_data=(X_val, y_val))

    # Get the best model's parameters
    model.load_weights(model_path)

    print("Evaluate model________________________")
    evaluate_model(model, history, y, X_test, y_test, model_dir)

    return model_dir


def build_svm(X, y):
    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = SVC()
    model.fit(X_train, y_train)

    training_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("-------------------------------------------")
    print("Training performance")
    print(classification_report(y_train, training_pred))
    print("-------------------------------------------")
    print("Testing performance")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-------------------------------------------")
