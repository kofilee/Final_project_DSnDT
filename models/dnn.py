import tensorflow as tf
from tensorflow.keras import Input, Model


class DNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, name="Dense")
        self.dense2 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu, name="Dense2")
        self.dense3 = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax, name="Softmax")

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output

    def build_graph(self):
        x = Input(shape=33)
        return Model(inputs=[x], outputs=self.call(x))
