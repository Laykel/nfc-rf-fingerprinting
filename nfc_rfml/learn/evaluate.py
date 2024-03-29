from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import classification_report, confusion_matrix

"""
Provide functions to save stats on a model and its performance and to plot confusion matrices and performance history.
"""


def write_stats(volume, segments_size, conf_mat, report, model_structure, output_dir):
    """Read information and statistics about our model and store it in a text file.
    :param volume: The volume of data of each class
    :param segments_size: The size of the signal segments used as input to the model
    :param conf_mat: The confusion matrix in text form
    :param report: The classification performance report
    :param model_structure: The description of the layers of the model
    :param output_dir: The folder where the file is to be written
    """
    amounts = f"Amount of samples for each class:\n{volume}"
    with open(output_dir / "stats.txt", "w") as file:
        file.write((f"{amounts}\nSegments size: {segments_size}\n\n{model_structure}\n\n"
                    f"{report}\n\nConfusion matrix:\n{conf_mat}"))


def plot_confusion_matrix(conf_mat, labels, output_dir):
    """Plot a nice looking confusion matrix and store it in an image file.
    :param conf_mat: The confusion matrix in text form
    :param labels: The labels of the classes
    :param output_dir: The folder where the file is to be written
    """
    conf_mat = conf_mat.astype(int)

    fig, ax = plt.subplots(figsize=(10, 10))
    conf_mat_image = ax.matshow(conf_mat, interpolation='nearest', cmap=cm.jet)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.annotate(str(conf_mat[i, j]), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14,
                        color="white",
                        weight="bold")

    ax.set_xticks(np.arange(conf_mat.shape[0]))
    ax.set_xticklabels([labels[lst] for lst in range(conf_mat.shape[0])], rotation='horizontal', fontsize=14)
    ax.set_yticks(np.arange(conf_mat.shape[1]))
    ax.set_yticklabels([labels[lst] for lst in range(conf_mat.shape[1])], fontsize=14)

    ax.set_xlabel('Predicted label', fontsize=16, labelpad=15)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Target label', fontsize=16, labelpad=15)

    fig.colorbar(conf_mat_image, orientation='vertical', pad=0.05)
    fig.savefig(output_dir / "conf_mat.png", bbox_inches='tight')
    plt.close(fig)


def plot_history(history, output_dir):
    """Plot the accuracy and the loss history during training and during validation and store those in an image file.
    :param history: The history object generated by Keras.
    :param output_dir: The folder where the file is to be written.
    """
    plt.rc('axes', axisbelow=True)
    # Plot accuracy history
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.grid(True)
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Training', 'Validation'], loc='upper left')
    fig.savefig(output_dir / "acc_history.png", bbox_inches='tight')
    plt.close(fig)

    # Plot loss history
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.grid(True)
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Training', 'Validation'], loc='upper left')
    fig.savefig(output_dir / "loss_history.png", bbox_inches='tight')
    plt.close(fig)


def plot_signal_window(window, index, label, output_dir):
    """Plot a signal window
    :param window: The window to plot
    :param index: The window index over the whole dataset
    :param label: The label to which the window actually belongs
    :param output_dir: The folder where our different files are to be written
    """
    fig, ax = plt.subplots(figsize=(30, 10))

    ax.plot(window[0], 'b-')
    ax.plot(window[1], 'r-')
    ax.grid(True)
    ax.set_title(f"Window index: {index}, Label: {label}")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend(["In phase", "In quadrature"], loc="upper left")

    fig.savefig(output_dir / f"t{label}-{index}.png", bbox_inches="tight")
    plt.close(fig)


def analyse_predictions(y, y_pred, X, output_dir):
    """Determine the wrongly predicted windows and plot them with some of the correctly predicted ones
    :param y: The complete labels set
    :param y_pred: The predicted labels
    :param X: The complete dataset
    :param output_dir: The folder where our different files are to be written
    """
    wrong_indices = np.nonzero((y_pred != y))[0]
    print("Wrong predictions:", wrong_indices)

    for index in wrong_indices:
        plot_signal_window(X[index], index, y[index], output_dir / "wrong-predictions")

    correct_indices = np.delete(np.arange(len(y)), wrong_indices)
    # Take as many correctly labelled windows as there are wrong ones
    selection = np.random.choice(correct_indices, len(wrong_indices), replace=False)

    for index in selection:
        plot_signal_window(X[index], index, y[index], output_dir / "correct-predictions")


def analyse_model(model, X, y, labels, output_dir):
    """Get the necessary data to analyse a reloaded model
    :param model: The classifier itself
    :param X: The test data
    :param y: The test labels
    :param labels: The labels for the whole dataset
    :param output_dir: The folder where our different files are to be written
    """
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    conf_mat = confusion_matrix(y, y_pred, labels=labels)
    plot_confusion_matrix(conf_mat, labels, output_dir)

    unique, counts = np.unique(y, return_counts=True)
    amounts = dict(zip(unique, counts))
    report = classification_report(y, y_pred)

    write_stats(amounts, len(X[0][0]), conf_mat, report, "", output_dir)

    analyse_predictions(y, y_pred, X, output_dir)


def evaluate_model(model, history, y, X_test, y_test, output_dir):
    """Get all the necessary data to fill our performance results folder and call appropriate functions.
    :param model: The classifier itself
    :param history: The history object returned after the training process
    :param y: The labels for the whole dataset
    :param X_test: The test data
    :param y_test: The test labels
    :param output_dir: The folder where our different files are to be written
    """
    # Evaluate model with test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Count the amount of data for each class
    unique, counts = np.unique(y, return_counts=True)
    amounts = dict(zip(unique, counts))

    # Get confusion matrix as well as the performance report
    labels = sorted(list(set(y_test)))
    conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred)

    # Read the model's summary
    structure = []
    model.summary(print_fn=lambda x: structure.append(x))
    model_structure = "\n".join(structure)

    write_stats(amounts, len(X_test[0][0]), conf_mat, report, model_structure, output_dir)

    plot_confusion_matrix(conf_mat, labels, output_dir)
    plot_history(history, output_dir)

    analyse_predictions(y_test, y_pred, X_test, output_dir)


def _test():
    cm = np.array([[0, 1], [2, 1]])
    plot_confusion_matrix(cm, ["cat", "dog"], Path('.'))

    class History:
        def __init__(self, history):
            self.history = history

    hist = {
        'loss': [0.5901445615434147, 0.5050772342918546, 0.5056924712380815,
                 0.49571301669640333, 0.49340261668077223],
        'accuracy': [0.6371653, 0.6592825, 0.6603407, 0.66175175, 0.6648206],
        'val_loss': [0.5044152915036236, 0.4941670439861439, 0.5019080130259196,
                     0.5011432712313569, 0.5064038568808709],
        'val_accuracy': [0.64308643, 0.6577778, 0.65197533, 0.657037, 0.6569136]
    }
    hist = History(hist)
    plot_history(hist, Path('.'))

    write_stats({0: 10000, 1: 12000, 2: 9000}, 512, cm,
                "rade\nasjdaiodj\nsadsd", "dasdasd\ndsad\ndsad",
                Path('.'))


if __name__ == "__main__":
    _test()
