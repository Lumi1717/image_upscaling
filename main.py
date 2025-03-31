import cv2

def upscale_image(input_path, output_path, scale_factor=2):
    # Read image
    img = cv2.imread(input_path)
    
    height, width = img.shape[:2]
    
    upscaled = cv2.resize(
        img, 
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_LANCZOS4
    )
    
    cv2.imwrite(output_path, upscaled)
    print(f"Upscaled image saved to {output_path}")

# Usage
upscale_image("images/low-res-72dpi.jpg", "images/output_2x.jpg", scale_factor=2)
