import tensorflow as tf
from level_2.models import generator, discriminator
from level_2.losses import esrgn_loss

class ESRGAN_Trainer:
    def __init__(self, hr_shape=(128, 128, 3), scale_factor=4):

        self.generator = generator.Generator(scale_factor=scale_factor)
        self.discriminator = discriminator.Discriminator()
        self.loss = esrgn_loss.ESRGAN_Loss()

        self.g_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4)

    
    @tf.function
    def train_step(self, lr_image, hr_image):
        with tf.GradientTape(persistent=True) as tape:
            sr_image = self.generator(lr_image, training = True)

            real_output = self.discriminator(hr_image, training = True)
            fake_output = self.discriminator(sr_image, training = True)

            # Calculator losses
            pixel_loss = self.loss.pixel_loss(hr= hr_image, sr= sr_image)
            percep_loss = self.loss.preceptual_loss(hr= hr_image, sr= sr_image)
            adv_loss = self.loss.relativistic_loss(real_output=real_output, fake_output=fake_output)

            g_loss = pixel_loss + 0.006 * percep_loss  + 0.001 * adv_loss
            d_loss = self.loss.adversarial_loss(real_output=real_output, fake_outout=fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        # Update discriminator 
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        return {'g_loss': g_loss, 'd_loss': d_loss}
    