import tensorflow as tf
from tensorflow.keras import layers, Model

class ESRGAN_Loss:
    def __init__(self):

        ## VGG19 for perceptual loss
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg = Model(vgg.input, vgg.layers[35].output)
        self.vgg.trainable = False

        self.l1_loss = tf.keras.losses.MeanAbsoluteError()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def pixel_loss(self, hr, sr):
        return self.l1_loss(hr, sr)
    
    def preceptual_loss(self, hr, sr):
        hr_features = self.vgg(hr)
        sr_features = self.vgg(sr)
        return self.l1_loss(hr_features, sr_features), 
    
    def adversarial_loss(self, real_output, fake_outout):
        real_loss = self.bce_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.bce_loss(tf.zeros_like(fake_outout), fake_outout)
        return( real_loss + fake_loss) * 0.5
    
    def relativistic_loss(self, real_output, fake_output):
        return self.bce_loss(
            tf.ones_like(real_output),
            tf.sigmoid(real_output - tf.reduce_mean(fake_output))
        )
    