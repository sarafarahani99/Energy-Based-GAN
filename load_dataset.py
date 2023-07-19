import tensorflow as tf


#load MNIST dataset
def prepare_dataset(batch_size, buffer_size):
    (train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    train_x = train_x / 255.0
    # train_x = (train_x-127.0) / 127.0
    train_images = (tf.data.Dataset.from_tensor_slices(train_x).shuffle(buffer_size)\
                    .batch(batch_size, drop_remainder=True).repeat())
    return train_images
