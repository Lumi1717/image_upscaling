import tensorflow as tf
from tensorflow.keras import layers, Model

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = tf.keras.Sequential([
            # input layer
            layers.Conv2D(64, kernel_size = 3, strides = 1, padding = 'same'),
            layers.LeakyReLU(alpha = 0.2),

            *self._discriminator_block(64, strides = 2),
            *self._discriminator_block(128, strides = 1),
            *self._discriminator_block(128, strides = 2),
            *self._discriminator_block(256, strides = 1),
            *self._discriminator_block(256, strides = 2),
            *self._discriminator_block(512, strides = 1),
            *self._discriminator_block(512, strides = 2),

            #Dense layer
            layers.Flatten(),
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1)
        ])

    def _discriminator_block(self, filters, strides):
        return [
            layers.Conv2D(filters, kernel_size = 3, strides=strides, padding = 'same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ] 
    
    def call(self, x):
        return self.model(x)