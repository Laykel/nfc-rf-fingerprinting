from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from .rfml_cnn import RFMLCNN


class YoussefCNN(RFMLCNN):
    """
    A Convolutional Neural network based on the one described in
    "Youssef et al. Machine Learning Approach to RF Transmitter Identification"
    """

    def __init__(self, nb_outputs, input_shape):
        super(YoussefCNN, self).__init__(nb_outputs, input_shape)

        self.conv1 = Conv2D(64, (2, 8), padding="same", activation="relu", input_shape=self.shape)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2 = Conv2D(32, 16, padding="same", activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(1, 2))

        self.flat = Flatten()
        self.dense = Dense(128, activation="relu")

        self.out = Dense(self.nb_outputs, activation="softmax")
        self.dropout = Dropout(0.5)

        self.build(self.shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        if training:
            x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.pool2(x)
        if training:
            x = self.dropout(x, training=training)

        x = self.flat(x)
        x = self.dense(x)
        return self.out(x)
