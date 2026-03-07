"""
Utility to extract individual face part images from a source image using binary masks.
Each part is cropped to its bounding box and saved as a separate image.
"""
import os
import cv2
import numpy as np
import argparse

def extract_parts(image_path, mask_dir, output_dir):
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Common region names
    regions = ["nose", "forehead", "left_cheek", "right_cheek", "chin"]
    
    for region in regions:
        # Look for mask file (handling potential prefixes)
        mask_path = None
        for f in os.listdir(mask_dir):
            if region in f and f.endswith(".png"):
                mask_path = os.path.join(mask_dir, f)
                break
        
        if not mask_path or not os.path.exists(mask_path):
            print(f"Warning: Mask for {region} not found in {mask_dir}")
            continue
            
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Ensure mask and image size match
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply mask (black out non-region area)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # Find bounding box of the mask to crop
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Crop to region
            cropped = masked_img[y:y+h, x:x+w]
            
            # Save
            out_path = os.path.join(output_dir, f"{region}_extracted.jpg")
            cv2.imwrite(out_path, cropped)
            print(f"Extracted {region} to {out_path}")
        else:
            print(f"Warning: Region {region} mask is empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    extract_parts(args.image, args.masks, args.output)
