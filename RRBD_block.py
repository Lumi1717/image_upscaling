import tensorflow as tf
import numpy as np
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
    

class Generator(Model):
    def __init__(self, scale_factor= 4):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor

        #Initial layers
        self.conv_first = layers.Conv2D(64, kernel_size = 3, padding = 'same')
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)


        #RRDB Block
        self.rrdb_blocks = [RRDB(64) for _ in range(23)]

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
        return 
    
    def adversarial_loss(self, real_output, fake_outout):
        real_loss = self.bce_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.bce_loss(tf.zeros_like(fake_outout), fake_outout)
        return( real_loss + fake_loss) * 0.5
    
    def relativistic_loss(self, real_output, fake_output):
        return self.bce_loss(
            tf.ones_like(real_output),
            tf.sigmoid(real_output - tf.reduce_mean(fake_output))
        )
    

class ESRGAN_Trainer:
    def __init__(self, hr_shape=(128, 128, 3), scale_factor=4):

        self.generator = Generator(scale_factor=scale_factor)
        self.discriminator = Discriminator()
        self.loss = ESRGAN_Loss()

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
    

trainer = ESRGAN_Trainer()

def train(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            lr_batch, hr_batch = batch
            losses = trainer.train_step(lr_batch, hr_batch)

        print (f"Epoch {epoch + 1}, G loss: {losses['g_loss']}, D loss: {losses['d_loss']:.4f}")

        if (epoch + 1) % 10 == 0:
            trainer.generator.save(f"esrgan_generator_epoch{epoch + 1}.h5")

def upscale_image(model, input_path, outoput_path, scale_factor =4):
    # load and process images
    lr_img = tf.io.read_file(input_path)
    lr_img = tf.image.decode_image(lr_img, channels=3)
    lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)
    lr_img = tf.expand_dims(lr_img, axis=0)

    #Generate High Res img

    sr_img = model(lr_img)
    sr_img = tf.clip_by_value(sr_img, 0, 1)
    sr_img = tf.image.convert_image_dtype(sr_img[0], tf.uint8)

    tf.io.write_file(outoput_path, tf.image.encode_jpeg(sr_img))
