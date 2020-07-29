from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from .rfml_cnn import RFMLCNN


# TODO l2 reg
class RiyazCNN(RFMLCNN):
    """
    A Convolutional Neural network based on the one described in
    "Riyaz et al. Deep Learning Convolutional Neural Networks for Radio Identification"
    """

    def __init__(self, nb_outputs, input_shape):
        super(RiyazCNN, self).__init__(nb_outputs, input_shape)

        self.conv1 = Conv2D(50, (1, 3), padding="same", activation="relu", input_shape=self.shape)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv2 = Conv2D(50, (2, 3), padding="same", activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(1, 2), strides=2)

        self.flat = Flatten()
        self.dense = Dense(256, activation="relu")

        self.out = Dense(self.nb_outputs, activation="softmax")
        self.dropout = Dropout(0.25)

        self.build(self.shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.pool2(x)

        x = self.flat(x)
        x = self.dense(x)
        if training:
            x = self.dropout(x, training=training)
        return self.out(x)
