import tensorflow as tf
from level_2.training import trainer

def train(dataset, epochs):
    trainer = trainer.ESRGAN_Trainer()

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

    #save output
    tf.io.write_file(outoput_path, tf.image.encode_jpeg(sr_img))
