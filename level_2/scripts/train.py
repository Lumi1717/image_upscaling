import yaml 
import tensorflow as tf
import os
from ..utils.data_loader import create_dataset
from ..training.trainer import ESRGAN_Trainer

def main():
    #load config

    with open("level_2/configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct absolute paths
    train_lr_dir = os.path.join(project_root, "/data/training/lr")
    train_hr_dir = os.path.join(project_root, "/data/training/hr/DIV2K_train_HR")
     
    # Print paths to verify
    print(f"Looking for LR training images in: {train_lr_dir}")
    print(f"Looking for HR training images in: {train_hr_dir}")
    
    # init training 
    train_dataset = create_dataset(
        lr_dir=train_lr_dir,
        hr_dir=train_hr_dir,
        batch_size=config["training"]["batch_size"],
        patch_size=config["training"]["patch_size"]
    )

    val_dataset = create_dataset(
        lr_dir=config["data"]["validation_lr_dir"],
        hr_dir=config["data"]["validation_hr_dir"],
        batch_size=4,
        patch_size=config["training"]["patch_size"]
    )

    trainer = ESRGAN_Trainer(config=config["training"])

    # Start training
    print("🚀 Starting Rocketship...")
    trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    # Enable mixed precision for faster training (if GPU supports it)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    main()