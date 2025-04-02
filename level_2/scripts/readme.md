# Image Upscaling Project

This project contains scripts for generating and training an ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model for image upscaling.


## Setup

1. Make sure you have all required dependencies installed:
```bash
pip install tensorflow opencv-python pyyaml tqdm matplotlib
```

2. Place your high-resolution training images in `level_2/data/training/hr/DIV2K_train_HR/`

## Usage

All commands should be run from the project root directory (`image_upscaling/`).

### 1. Generate Low-Resolution Images
Generate low-resolution images from your high-resolution dataset:
```bash
python -m level_2.scripts.generate_lr_images
```
This script reads settings from `configs/train_config.yaml` and processes images in parallel.

### 2. Verify Generated Images
Check that the LR images were generated correctly by viewing HR/LR pairs:
```bash
python -m level_2.scripts.verification_script
```
This will display a sample pair of high-resolution and corresponding low-resolution images.

### 3. Train the Model
Start the ESRGAN training:
```bash
python -m level_2.scripts.train
```
Training parameters can be configured in `configs/train_config.yaml`.

### 4. Run Inference
Upscale new images using the trained model:
```bash
python -m level_2.scripts.inference
```

## Configuration

The `train_config.yaml` file contains important settings:
```yaml
data:
  train_lr_dir: "level_2/data/training/lr"
  train_hr_dir: "level_2/data/training/hr/DIV2K_train_HR"
  val_lr_dir: "level_2/data/val/lr"
  val_hr_dir: "level_2/data/val/hr"
  scale: 4          # Upscaling factor
  workers: 8        # Number of parallel workers for data generation

training:
  batch_size: 16
  patch_size: 128
  epochs: 1000
  learning_rate: 1e-4
  # ... other training parameters
```

## Troubleshooting

1. If you get import errors, make sure you're:
   - Running scripts from the project root directory
   - Using the `-m` flag to run scripts as modules
   - Have all `__init__.py` files in place

2. If images aren't loading:
   - Verify file paths in `train_config.yaml`
   - Check file permissions
   - Ensure images are in supported formats (PNG, JPEG, JPG)

3. For CUDA/GPU issues:
   - Check TensorFlow GPU support is properly installed
   - Monitor GPU memory usage during training


