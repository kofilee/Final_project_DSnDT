import tensorflow as tf
from tensorflow.keras import Input, Model


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=32,  # 卷積層神經元（卷積核）數目
            kernel_size=[2, 2],  # 接受區的大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu,  # 激活函数
        )
        self.pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1, name="MaxPool")
        self.flatten = tf.keras.layers.Flatten(name="Flatten")
        self.dense = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax, name="Softmax")

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output

    def build_graph(self):
        x = Input(shape=(11, 3, 1))
        return Model(inputs=[x], outputs=self.call(x))
