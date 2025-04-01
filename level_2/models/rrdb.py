import tensorflow as tf
from tensorflow.keras import layers, Model

class RRDB(layers.Layer):

    def __init__(self, filters, beta=0.2):
        super(RRDB, self).__init__()
        self.beta = beta

        self.conv1 = layers.Conv2D (filters, kernel_size = 3, padding ='same')
        self.conv2 = layers.Conv2D (filters, kernel_size = 3, padding = 'same')
        self.conv3 = layers.Conv2D (filters, kernel_size = 3, padding = 'same')

        self.leaky_relu  = layers.LeakyReLU()
    
    def call(self, x):
        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(x))
        out3  = self.leaky_relu(self.conv3(x))

        return x + self.beta * out3
    


