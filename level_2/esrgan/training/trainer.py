import tensorflow as tf
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from level_2.esrgan.models import generator, discriminator
from level_2.esrgan.losses import esrgn_loss

class ESRGAN_Trainer:
    def __init__(self, config=None, hr_shape=(128, 128, 3), scale_factor=4):

        self.config = config if config else{
            'epochs': 1000,
            'save_intervals': 10,
            'sample_intervals':10,
            'validation_intervals':10
        }

        self.generator = generator.Generator(scale_factor=scale_factor)
        self.discriminator = discriminator.Discriminator()
        self.loss = esrgn_loss.ESRGAN_Loss()

        self.g_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = Path("checkpoint")
        self.sample_dir = Path("sample")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
    
    @tf.function
    def train_step(self, lr_image, hr_image):
        with tf.GradientTape(persistent=True) as tape:
            sr_image = self.generator(lr_image, training = True)

            real_output = self.discriminator(hr_image, training = True)
            fake_output = self.discriminator(sr_image, training = True)

            # Calculator losses
            pixel_loss = self.loss.pixel_loss(hr= hr_image, sr= sr_image)
            percep_loss = self.loss.perceptual_loss(hr= hr_image, sr= sr_image)
            adv_loss = self.loss.relativistic_loss(real_output=real_output, fake_output=fake_output)

            g_loss = pixel_loss + 0.006 * percep_loss  + 0.001 * adv_loss
            d_loss = self.loss.adversarial_loss(real_output=real_output, fake_output=fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        # Update discriminator 
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        return {'g_loss': g_loss, 'd_loss': d_loss, 'pixel_loss': pixel_loss, 'adv_loss': adv_loss}
    
    def _validate(self, val_dataset):
        psnr_matric = tf.keras.metrics.Mean()
        ssim_matric = tf.keras.metrics.Mean()

        for lr, hr in val_dataset:
            sr = self.generator(lr, training=False)
            psnr_matric.update_state(tf.image.psnr(hr, sr, max_val=1.0))
            ssim_matric.update_state(tf.image.ssim(hr, sr, max_val=1.0))
        
        return {
            'psnr': psnr_matric.result().numpy(),
            'ssim': ssim_matric.result().numpy()
        }
    
    def _save_samples(self, epoch, lr, hr, sr, num_samples=3):
        os.makedirs(self.sample_dir, exist_ok=True)

        #convert tensors to numpy
        lr = np.clip(lr.numpy(), 0, 1)
        hr = np.clip(hr.numpy(), 0, 1)
        sr = np.clip(sr.numpy(), 0, 1)

        # visalzie 
        plt.figure(figsize=(15, 5*num_samples))

        for i in range(min(num_samples, lr.shape[0])):
            # Plot LR image
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(lr[i])
            plt.title(f"LR Input\n{lr[i].shape[:2]} → {hr[i].shape[:2]}")
            plt.axis('off')
            
            # Plot SR image
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(sr[i])
            plt.title(f"Super Resolved\nPSNR: {tf.image.psnr(hr[i], sr[i], max_val=1.0):.2f}")
            plt.axis('off')
            
            # Plot HR image
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(hr[i])
            plt.title("Original HR")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.sample_dir, f"compare_epoch_{epoch+1}.png"))
        plt.close()
    
    def _save_checkpoint(self, epoch):
        generator_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch+1}.h5")
        self.generator.save(generator_path)
        
        # Save discriminator
        discriminator_path = os.path.join(self.checkpoint_dir, f"discriminator_epoch_{epoch+1}.h5")
        self.discriminator.save(discriminator_path)
        
        # Keep only the latest 3 checkpoints
        self._cleanup_checkpoints(keep=3)
    


    def _cleanup_checkpoints(self, keep=3):
        """Maintain only the most recent checkpoints"""
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("generator_epoch_")],
            key=lambda x: int(x.split("_")[2].split(".")[0])
        )
        
        # Remove older checkpoints
        for old_checkpoint in checkpoints[:-keep]:
            os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
            os.remove(os.path.join(self.checkpoint_dir, old_checkpoint.replace("generator", "discriminator")))



    def train(self, train_dataset, val_dataset):
        summary_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        for epoch in range(self.config['epochs']):
            for step, (lr, hr) in enumerate(train_dataset):
                losses = self.train_step(lr, hr)
                if step % 100 == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('generator_loss', losses['g_loss'], step=step)
                        tf.summary.scalar('discriminator_loss', losses['d_loss'], step=step)

                if (epoch + 1) % self.config['sample_intervals'] == 0:
                    val_metrics = self._validate(val_dataset=val_dataset)
                    print(f"Validation Pero PSNR: {val_metrics['psnr']:.2f}, Validation Pero SSIM: {val_metrics['ssim']:.4f}")

                if (epoch + 1) % self.config['checkpoint_intervals'] == 0:
                    self._save_checkpoint(epoch=epoch)

            # Validation and samples
            if epoch % 10 == 0:
                self._validate(val_dataset)
                self._save_samples(epoch)
                self._save_checkpoint(epoch)
    