import tensorflow as tf
from tensorflow.keras import Input, Model


class RNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        GRU_cell = tf.keras.layers.GRUCell(
            11,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            dropout=0.5,
            bias_initializer='zeros',
        )
        self.gru = tf.keras.layers.RNN(GRU_cell,
                                       return_sequences=True,
                                       name='GRU')
        self.flatten = tf.keras.layers.Flatten(name="Flatten")
        self.dense = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax, name="Softmax")

    def call(self, inputs, **kwargs):
        x = self.gru(inputs)
        x = self.flatten(x)
        output = self.dense(x)
        return output

    def build_graph(self):
        x = Input(shape=(11, 3))
        return Model(inputs=[x], outputs=self.call(x))
