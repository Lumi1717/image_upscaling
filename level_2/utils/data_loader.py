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
    def preprocess_image(lr_path, hr_path):
        lr = tf.image.decode_image(tf.io.read_file(lr_path), dtype=tf.float32)
        hr = tf.image.decode_image(tf.io.read_file(hr_path), dtype=tf.float32)
        

        # random chopping
        shape = tf.shape(hr)
        hr_patch = tf.image.random_crop(hr, [patch_size, patch_size, 3])
        lr_patch = tf.image.resize(hr_patch, [patch_size//4 , patch_size//4])

        # Random flips
        if tf.random.uniform(()) > 0.5:
            lr_patch = tf.image.flip_left_right(lr_patch)
            hr_patch = tf.image.flip_left_right(hr_patch)
        return lr_patch, hr_patch
    
    lr_files = sorted(glob.glob(f"{lr_dir}"))
    hr_files = sorted(glob.glob(f"{hr_dir}"))
    
    dataset = tf.data.Dataset.from_tensor_slices((lr_files, hr_files))
    dataset = dataset.shuffle(1000).map(preprocess_image, num_parallel_calls= tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    