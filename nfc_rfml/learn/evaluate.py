import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def write_stats(volume, conf_mat, report, model_structure, output_dir):
    """

    :param volume:
    :param conf_mat:
    :param report:
    :param model_structure:
    :param output_dir:
    :return:
    """
    amounts = f"Amount of samples for each class:\n{volume}"
    with open(f"{output_dir}/stats.txt", "w") as file:
        file.write(f"{amounts}\n\n{model_structure}\n\n{report}\n\nConfusion matrix:\n{conf_mat}")


def plot_confusion_matrix(conf_mat, labels, output_dir):
    """

    :param conf_mat:
    :param labels:
    :param output_dir:
    :return:
    """
    conf_mat = conf_mat.astype(int)

    fig, ax = plt.subplots()
    conf_mat_image = ax.matshow(conf_mat, interpolation='nearest', cmap=cm.jet)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.annotate(str(conf_mat[i, j]), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12,
                        color="darkgrey",
                        weight="bold")

    ax.set_xticks(np.arange(conf_mat.shape[0]))
    ax.set_xticklabels([labels[lst] for lst in range(conf_mat.shape[0])], rotation='horizontal')
    ax.set_yticks(np.arange(conf_mat.shape[1]))
    ax.set_yticklabels([labels[lst] for lst in range(conf_mat.shape[1])])

    ax.set_xlabel('Predicted label')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Target label')

    fig.colorbar(conf_mat_image, orientation='vertical', pad=0.05)
    fig.savefig(f"{output_dir}/conf_mat.png", bbox_inches='tight')
    plt.close(fig)


def plot_history(history, output_dir):
    """

    :param history:
    :param output_dir:
    :return:
    """
    # Plot accuracy history
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(f"{output_dir}/acc_history.png", bbox_inches='tight')
    plt.close(fig)

    # Plot loss history
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(f"{output_dir}/loss_history.png", bbox_inches='tight')
    plt.close(fig)


def _test():
    cm = np.array([[0, 1], [2, 1]])
    plot_confusion_matrix(cm, ["cat", "dog"], ".")

    class History:
        def __init__(self, history):
            self.history = history

    history = {
        'loss': [0.5901445615434147, 0.5050772342918546, 0.5056924712380815,
                 0.49571301669640333, 0.49340261668077223],
        'accuracy': [0.6371653, 0.6592825, 0.6603407, 0.66175175, 0.6648206],
        'val_loss': [0.5044152915036236, 0.4941670439861439, 0.5019080130259196,
                     0.5011432712313569, 0.5064038568808709],
        'val_accuracy': [0.64308643, 0.6577778, 0.65197533, 0.657037, 0.6569136]
    }
    hist = History(history)
    plot_history(hist, ".")

    write_stats({0: 10000, 1: 12000, 2: 9000}, cm, "rade\nasjdaiodj\nsadsd", "dasdasd\ndsad\ndsad", ".")


if __name__ == "__main__":
    _test()
