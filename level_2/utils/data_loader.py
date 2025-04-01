import glob
import tensorflow as tf

def create_dataset(lr_dir, hr_dir, batch_size=18, patch_size = 64):
    """Creates a tf.data.Dataset for ESRGAN training.
    
    Args:
        lr_dir: Path to low-resolution images
        hr_dir: Path to high-resolution images
        batch_size: Batch size
        patch_size: Size of random crops
        
    Returns:
        Configured tf.data.Dataset
    """

    # Get list of image files
    lr_files = tf.data.Dataset.list_files(lr_dir + "/*", shuffle=True)
    hr_files = tf.data.Dataset.list_files(hr_dir + "/*", shuffle=True)
    
    # Zip the two datasets together
    dataset = tf.data.Dataset.zip((lr_files, hr_files))

    def preprocess_image(lr_path, hr_path):
        # Read images from files
        lr = tf.io.read_file(lr_path)
        hr = tf.io.read_file(hr_path)
        
        
         # Decode images
        lr = tf.image.decode_image(lr, channels=3)
        hr = tf.image.decode_image(hr, channels=3)

        # Convert to float32 and normalize to [0, 1]
        lr = tf.cast(lr, tf.float32) / 255.0
        hr = tf.cast(hr, tf.float32) / 255.0

         # Ensure shapes are set
        lr.set_shape([None, None, 3])
        hr.set_shape([None, None, 3])

    
        # Random crop if patch_size is specified
        if patch_size:
            lr = tf.image.random_crop(lr, [patch_size, patch_size, 3])
            hr = tf.image.random_crop(hr, [patch_size * 4, patch_size * 4, 3])  # assuming 4x upscaling
            
        return lr, hr
    
    lr_files = sorted(glob.glob(f"{lr_dir}"))
    hr_files = sorted(glob.glob(f"{hr_dir}"))
    
    dataset = tf.data.Dataset.from_tensor_slices((lr_files, hr_files))
    dataset = dataset.shuffle(1000).map(preprocess_image, num_parallel_calls= tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    

def augment_image(lr_patch, hr_patch):
    # rand rotations

    if tf.random.uniform(()) > 0.5:
        k = tf.random(shape = [], minival= 0, maxval=4, dtype=tf.int32)
        lr_patch = tf.image.rot90(lr_patch, k=k)
        hr_patch = tf.image.rot90(hr_patch, k=k)
    
    lr_patch = tf.image.random_brightness(lr_patch, 0.1)
    hr_patch = tf.image.random_brightness(hr_patch, 0.1)

    return lr_patch, hr_patch