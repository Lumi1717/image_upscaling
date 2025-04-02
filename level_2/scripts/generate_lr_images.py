import os
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def downscale_image (hr_path,lr_dir, scale_factor =4):

    hr_img = cv2.imread(hr_path)
    if hr_img is None:
        print(f"DID NOT READ {hr_path} fix yo shit")
        return
    
    #ceate LR imgsss
    h, w = hr_img.shape[:2]
    lr_img = cv2.resize(hr_img, (w//scale_factor, h//scale_factor), interpolation= cv2.INTER_CUBIC )

    #save imgs
    file_name = os.path.basename(hr_path)
    lr_path = os.path.join(lr_dir, file_name)
    cv2.imwrite(lr_path, lr_img)

def process_args(args):
    downscale_image(*args)

def generate_lr_imgs(hr_dir, lr_dir, scale_factor=4, num_workers =4):

    os.makedirs(lr_dir, exist_ok=True)

    hr_img = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
                if f.lower().endswith(('.png', '.jpeg', 'jpg'))]
    
    task_args = [(hr_path, lr_dir, scale_factor) for hr_path in hr_img]

        # Process with multiprocessing
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_args, task_args), 
                total=len(hr_img), 
                desc="Generating LR images"))
    
    print(f"Generated {len(hr_img)} LR images in {lr_dir}")

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))

if __name__ == '__main__':

    project_root = get_project_root()
    
    with open(os.path.join(project_root, "level_2/configs/train_config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    hr_dir = os.path.join(project_root, config['data']['train_hr_dir'])
    lr_dir = os.path.join(project_root, config['data']['train_lr_dir'])
    scale = config['data']['scale_factor']
    workers = config['data']['num_workers']

    print(f"Processing images from: {hr_dir}")
    print(f"Saving results to: {lr_dir}")
  
    generate_lr_imgs(hr_dir, lr_dir, scale, workers)
