from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout


class RFMLCNN(Model):
    """
    Simple standard keras Model extended to make the summary work and parameterize the input shape
    and the number of outputs.
    """

    def __init__(self, nb_outputs, input_shape):
        super(RFMLCNN, self).__init__()
        self.nb_outputs = nb_outputs
        self.shape = input_shape

    def summary(self, **kwargs):
        """
        Override summary method in order to first build a model.
        This allows the output shapes to be correctly shown.
        """
        x = Input(shape=self.shape[1:])
        Model(inputs=[x], outputs=self.call(x)).summary(**kwargs)
