import tensorflow as tf

def create_dataset(lr_dir, hr_dir, batch_size=18, patch_size = 64):
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