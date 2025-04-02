import cv2
import yaml
import os
import matplotlib.pyplot as plt
from level_2.scripts.generate_lr_images import get_project_root

project_root = get_project_root()

# Print current working directory to debug
print(f"Current working directory: {os.getcwd()}")

with open(os.path.join(project_root, "level_2/configs/train_config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

hr_dir = os.path.join(project_root, config['data']['train_hr_dir'], '0001.png')
lr_dir = os.path.join(project_root, config['data']['train_lr_dir'], '0001.png')  # Fixed 'lt' to 'lr'

hr = cv2.imread(hr_dir)
lr = cv2.imread(lr_dir)

plt.figure(figsize=(10,5))
plt.subplot(121); plt.imshow(hr); plt.title("HR")
plt.subplot(122); plt.imshow(lr); plt.title("LR")
plt.show()
