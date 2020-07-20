import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocess.format import split_data
from learn.models import rfmlcnn

"""
...
"""


def build_cnn(X, y, epochs):
    # Split data into train, validation and test data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, 0.7, 0.2, 0.1)

    # Build model and output its structure
    shape = (None,) + X_train.shape[1:]
    model = rfmlcnn.RFMLCNN(nb_outputs=len(set(y)), input_shape=shape)
    model.summary()

    # Configure model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Make sure the training stops when the performance stops getting better
    # and save the best model to disk
    callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                 ModelCheckpoint(filepath='test.tf', monitor='val_loss', save_best_only=True)]

    # TODO Cross validation?
    # Train model and adjust with validation set
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val))
    print("History:", history.history)

    # -------------------------------------------------------------------------------------------
    # TODO put that in an evaluate module

    model.load_weights("test.tf")

    # Evaluate model with test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


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
