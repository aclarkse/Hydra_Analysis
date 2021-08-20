import os
import sys
import glob
import numpy as np
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
import cv2
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt


def make_patches(image_path, mask_path):
  imgs = glob.glob(os.path.join(image_path, '*.tif'))
  mks = glob.glob(os.path.join(mask_path, '*.tif'))
  print(f"Found {len(imgs)} images and masks.")
  sys.stdout.flush()
  
  print("Processing images...")
  sys.stdout.flush()
  for img in tqdm(imgs, total=len(imgs)):
      filename = img.split('/')[3].split('.')[0]
      large_image = tiff.imread(img)
      patches = patchify(large_image, (256, 256), step=256)

      # save the image patches
      img_patches_dir = 'image_patches'
      for i in range(patches.shape[0]):
          for j in range(patches.shape[1]):
              single_patch = patches[i, j, :, :]
              try:
                  os.makedirs(img_patches_dir, exist_ok=True)
                  tiff.imwrite(os.path.join(img_patches_dir,
                               f'{filename}_patch_{i}_{j}'), single_patch)
              except OSError as error:
                  print(f'Directory \'{img_patches_dir}\' cannot be created!')
                  
  print("Processing masks...")
  sys.stdout.flush()
  for mask in tqdm(mks, total=len(mks)):
      filename = mask.split('/')[3].split('.')[0]
      large_mask = tiff.imread(mask)
      patches = patchify(large_mask, (256, 256), step=256)
      
      # save the image patches
      mask_patches_dir = 'mask_patches'
      for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            try:
                os.makedirs(mask_patches_dir, exist_ok = True)
                tiff.imwrite(os.path.join(mask_patches_dir,
                            f'{filename}_patch_{i}_{j}'), single_patch)
            except OSError as error:
                print(f'Directory \'{mask_patches_dir}\' cannot be created!')
                
  print('Finished processing!')
  
if __name__ == '__main__':
    image_path = 'hydra/train/images/'
    mask_path = 'hydra/train/masks/'

    make_patches(image_path, mask_path)

