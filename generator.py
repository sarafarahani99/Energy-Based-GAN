import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, is_training=True, batch_size=64):
        super(Generator, self).__init__(name="generator")
        self.is_training = is_training
        self.batch_size = batch_size

        self.dens_1 = tf.keras.layers.Dense(1024)
        self.dense_2 = tf.keras.layers.Dense(128 * 7 * 7)

        self.deConv2d_1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                          kernel_size=4,
                                                          strides=2,
                                                          padding='same',
                                                          kernel_initializer=tf.keras.initializers.RandomNormal(
                                                              stddev=0.02))

        self.deConv2d_2 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same',
                                                          activation="sigmoid")

        self.bn_1 = tf.keras.layers.BatchNormalization(trainable=self.is_training)
        self.bn_2 = tf.keras.layers.BatchNormalization(trainable=self.is_training)
        self.bn_3 = tf.keras.layers.BatchNormalization(trainable=self.is_training)

    def call(self, inputs, training):
        x = self.dens_1(inputs)
        x = self.bn_1(x, training)
        x = tf.keras.layers.ReLU()(x)
        x = self.dense_2(x)
        x = self.bn_2(x, training)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((7, 7, 128))(x)
        x = self.deConv2d_1(x)
        x = self.bn_3(x, training)
        x = tf.keras.layers.ReLU()(x)
        x = self.deConv2d_2(x)
        return x
