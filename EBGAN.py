import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import generator
import discriminator
import load_dataset


def get_noise_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


def discriminator_loss(d_real_img_err, d_fake_img_err, margin=16):
    return d_real_img_err + tf.maximum(margin - d_fake_img_err, 0)


def generator_loss(d_fake_img_err, encoder_output):
    encoder_output_norm = tf.sqrt(tf.reduce_sum(tf.square(encoder_output), 1, keepdims=True))
    normalized_term = encoder_output / encoder_output_norm
    pt_term = tf.matmul(normalized_term, normalized_term, transpose_b=True)
    pt_term = tf.reduce_sum(pt_term)
    pt_loss = pt_term / (batch_size * (batch_size - 1))
    return d_fake_img_err + pt_loss


# one train step
@tf.function
def train_step(real_images):
    z_sample = get_noise_z(z_noise_dimension, batch_size)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        generator_fake_images = generator(z_sample, training=True)

        d_fake_img_err, fake_img_encoder_output = discriminator(generator_fake_images, training=True)
        d_real_img_err, _ = discriminator(real_images, training=True)

        d_loss = discriminator_loss(d_real_img_err, d_fake_img_err)
        g_loss = generator_loss(d_fake_img_err, fake_img_encoder_output)

    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return g_loss, d_loss


# training loop
def train_EBGAN(dataset, summary_writer_filepath):
    real_images = iter(dataset)
    summary_writer = tf.summary.create_file_writer(summary_writer_filepath)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               g_optimizer=generator_optimizer,
                               d_optimizer=discriminator_optimizer,
                               g=generator,
                               d=discriminator)
    ckpt_mngr = tf.train.CheckpointManager(ckpt, summary_writer_filepath, max_to_keep=8)
    # ckpt.restore(ckpt_mngr.latest_checkpoint)
    for step in range(iterations):
        ckpt.step.assign_add(1)
        real_images_batch = next(real_images)
        g_loss, d_loss = train_step(real_images_batch)

        generator_loss_metric(g_loss)
        discriminator_loss_metric(d_loss)
        total_loss_metrics(g_loss + d_loss)

        if step % 100 == 0:
            print("Discriminator loss at step {} is :".format(step), discriminator_loss_metric.result())
            print("Generator loss at step {} is :".format(step), generator_loss_metric.result())
            print("Total loss at step {} is :".format(step), total_loss_metrics.result())
            save_path = ckpt_mngr.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            with summary_writer.as_default():
                with tf.summary.record_if(lambda: step % 100 == 0):
                    tf.summary.scalar('generator_loss', generator_loss_metric.result(), step=step)
                    tf.summary.scalar('discriminator_loss', discriminator_loss_metric.result(), step=step)
            generator_loss_metric.reset_states()
            discriminator_loss_metric.reset_states()
            total_loss_metrics.reset_states()
            # Plot 5 sample generated images
            for i in range(6):
                plt.subplot(1, 6, i + 1)
                I = generator(get_noise_z(z_noise_dimension, batch_size))[i, :, :]
                plt.imshow(I)
                plt.axis('off')
            plt.show()


if __name__ == "__main__":
    summary_writer_filepath = "summary"
    iterations = 20000
    batch_size = 512
    # buffer_size is set to number of mnist training images
    buffer_size = 60000
    input_shape = (28, 28, 1)
    random_seed = 42
    z_noise_dimension = 100

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    train_images = load_dataset.prepare_dataset(batch_size, buffer_size)

    # define generator and discriminator models
    generator = generator.Generator((z_noise_dimension,))
    discriminator = discriminator.Discriminator(input_shape)

    # define generator and discriminator optimizers
    generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)

    # define loss metrics
    generator_loss_metric = tf.metrics.Mean(name='generator_loss')
    discriminator_loss_metric = tf.metrics.Mean(name='discriminator_loss')
    total_loss_metrics = tf.metrics.Mean(name='total_loss')

    train_EBGAN(train_images, summary_writer_filepath)