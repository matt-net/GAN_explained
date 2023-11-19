from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import BinaryCrossentropy
# from GAN_network import GenAI_network as network
import tensorflow as tf


class FashionGAN(Model):
    def __init__(self, generator, discriminator, g_lr=0.001, d_lr=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_lr = g_lr
        self.d_lr = d_lr

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.g_opt = adam_v2.Adam(learning_rate=self.g_lr)
        self.d_opt = adam_v2.Adam(learning_rate=self.d_lr)
        self.g_loss = BinaryCrossentropy()
        self.d_loss = BinaryCrossentropy()

    def train_step(self, batch):
        real_images = batch  # get data
        fake_images = self.generator(tf.random.normal((128, 128)))

        # Train Discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake data to the discriminator
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create label for read & fake data. 0 : REAL , 1 : FAKE
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the True output
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

            # Apply backpropagation
            dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128, 128)), training=True)

            # Create the predicted label
            predicted_label = self.discriminator(gen_images, training=False)

            # Calculate loss
            total_g_loss = self.g_loss(tf.zeros_like(predicted_label), predicted_label)

            # Apply backpropagation - trick to training to fake out discriminator
            ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}
