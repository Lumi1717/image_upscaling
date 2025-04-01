import yaml 
import tensorflow as tf

from esrgan.utils.data_loader import create_dataset
from esrgan.training.trainer import ESRGAN_Trainer

def main():
    #load config

    with open("configs/train_config.yaml") as f:
        config = yaml.safe_dump(f)
    
    # init training 

    train_dataset = create_dataset(
        lr_dir=config["data"]["train_lr_dir"],
        hr_dir=config["data"]["train_hr_dir"],
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