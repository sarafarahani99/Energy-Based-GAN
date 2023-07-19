import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, is_training=True, batch_size=64):
        super(Discriminator, self).__init__(name="discriminator")
        self.is_training = is_training
        self.batch_size = batch_size

        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

        self.deConv2d_1 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4,
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                                                 activation="sigmoid")

        self.bn_1 = tf.keras.layers.BatchNormalization(trainable=self.is_training, epsilon=1e-5)

        self.bn_2 = tf.keras.layers.BatchNormalization(trainable=self.is_training, epsilon=1e-5)

        self.fc_1 = tf.keras.layers.Dense(units=32, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

        self.fc_2 = tf.keras.layers.Dense(units=64 * 14 * 14, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, inputs, training):
        # encoder
        x = self.conv_1(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x_encoder_output = self.fc_1(x)
        # decoder
        x = self.fc_2(x_encoder_output)
        x = self.bn_1(x, training)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((14, 14, 64))(x)
        x = self.deConv2d_1(x)
        d_error = tf.math.sqrt(2 * tf.nn.l2_loss(x - inputs)) / self.batch_size
        return d_error, x_encoder_output

