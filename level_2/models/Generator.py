import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from level_2.models import rrdb


class Generator(Model):
    def __init__(self, scale_factor= 4):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor

        #Initial layers
        self.conv_first = layers.Conv2D(64, kernel_size = 3, padding = 'same')
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)


        #RRDB Block
        self.rrdb_blocks = [rrdb.RRDB(64) for _ in range(23)]

        self.truck_conv = layers.Conv2D(64, kernel_size = 3, padding = 'same')

        self.upsampling = []

        for _ in range(int(np.log2(scale_factor))):
            self.upsampling.extend([
                layers.Conv2D(64*4, kernel_size = 3, padding = 'same'),
                layers.Conv2D(lambda x: tf.nn.depth_to_space(x, 2)),
                layers.LeakyReLU(alpha=0.2)
            ])

        # Final output
        self.conv_last = layers.Conv2D(3, kernel_size = 3, padding = 'same')
    
    def call(self, x):
        fea = self.leaky_relu(self.conv_first(x))
        trunk = fea

        for block in self.rrdb_blocks:
            trunk = block(trunk)
        
        trunk = self.truck_conv(trunk)
        fea = fea + trunk

        #Upsampling

        for layer in self.upsampling:
            fea = layer(fea)

        out = self.conv_last(fea)
        return out