import tensorflow as tf
import argparse
from pathlib import Path
from level_2.esrgan.models.generator import Generator

def upscale_img (model_path, input_path, output_path, scale_factor = 4):

    #load model
    generator = tf.keras.models.load_model(model_path, compile=False) 


    #read and process the imahes
    lr_img = tf.io.read_file(input_path)
    lr_img = tf.image.decode_image(lr_img, channels=3, expand_animations=False)
    lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)
    lr_img = tf.expand_dims(lr_img, axis=0)

    #generate SR images

    sr_img = generator(lr_img)
    sr_img = tf.clip_by_value(sr_img, 0, 1)
    sr_img = tf.image.convert_image_dtype(sr_img[0], tf.uint8)

    #save sr images
    tf.io.write_file(output_path, tf.image.encode_jpeg(sr_img))


if __name__== 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to generator .h5 file")
    parser.add_argument("--input", type=str, required=True,  help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--scale", type=str, default=4, help="Upscaling factor")
    args = parser.parse_args()


    print(f"🔍 Upscaling {args.input} (x{args.scale})...")
    upscale_img(args.model, args.input, args.output, args.scale)
    print(f"✅ Saved to {args.output}")