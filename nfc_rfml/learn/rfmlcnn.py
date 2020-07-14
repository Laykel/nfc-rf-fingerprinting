from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


"""
youssef
"""


class RFMLCNN(Model):
    # TODO see if I can add methods for results (report, conf_mat...)
    def __init__(self, nb_outputs, input_shape):
        super(RFMLCNN, self).__init__()
        self.inp = Conv2D(64, kernel_size=(2, 8), padding="same", activation="relu", input_shape=input_shape)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(32, 16, padding="same", activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(1, 2))
        self.flat = Flatten()
        self.dense = Dense(128, activation="relu")
        self.outp = Dense(nb_outputs, activation="sigmoid")

        self.build(input_shape)

    def call(self, inputs, training=False):
        # TODO dropout in case of training
        x = self.inp(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.dense(x)
        return self.outp(x)
