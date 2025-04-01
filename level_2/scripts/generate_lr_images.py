import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def downscale_image (lr_path, hr_path, scale_factor =4):

    hr_img = cv2.imread(hr_path)
    if hr_img is None:
        print("DID NOT READ {hr_path} fix yo shit")
        return
    
    #ceate LR imgsss

    h, w = hr_img.shape[:2]
    lr_img = cv2.resize(hr_img, (w//scale_factor, h//scale_factor), interpolation= cv2.INTER_CUBIC )

    #save imgs

    file_name = os.path.basename(hr_path)
    lr_path = os.path.join(lr_path, file_name)
    cv2.imread(lr_path, lr_img)

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", required=True, help="Path to HR images")
    parser.add_argument("--lr_dir", required=True, help="Output path for LR images")
    parser.add_argument("--scale", type=int, default=4, help="Downscaling factor")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    generate_lr_imgs(args.hr_dir, args.lr_dir, args.scale, args.workers)

#     cd /Users/ahlamyusuf/Documents/image_upscaling/level_2
# python scripts/generate_lr_images.py \
#     --hr_dir data/training/hr/DIV2K_train_HR \
#     --lr_dir data/lr/training \
#     --scale 4 \
#     --workers 8