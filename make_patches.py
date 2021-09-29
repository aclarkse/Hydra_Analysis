import os
import shutil
import sys
import glob
import tifffile as tiff
from tqdm import tqdm
from patchify import patchify


def make_patches(image_path, mask_path, patch_size):
  img_stack = tiff.imread(image_path)
  msk_stack = tiff.imread(mask_path)
  sys.stdout.flush()

  print("Processing images...")
  sys.stdout.flush()
  num_images = img_stack.shape[0]
  for img in tqdm(range(num_images)):
    large_img = img_stack[img]
    img_patches = patchify(large_img, (patch_size, patch_size), step=patch_size)
    
    for i in range(img_patches.shape[0]):
      for j in range(img_patches.shape[1]):
        single_patch = img_patches[i,j,:,:]
        imgs_dir = 'patches/images'
        os.makedirs(imgs_dir, exist_ok=True)
        tiff.imwrite(os.path.join(imgs_dir, f'patch_{i}_{j}.tif'), single_patch)
              
  print('Processing masks...')
  sys.stdout.flush()
  num_masks = msk_stack.shape[0]
  for msk in tqdm(range(num_masks)):
      large_msk = msk_stack[msk]
      msk_patches = patchify(large_msk, (patch_size, patch_size), step=patch_size)
      
      for i in range(msk_patches.shape[0]):
        for j in range(msk_patches.shape[1]):
          single_patch = msk_patches[i,j,:,:]
          single_patch = single_patch / 255.
          mks_dir = 'patches/masks'
          os.makedirs(mks_dir, exist_ok=True)
          tiff.imwrite(os.path.join(mks_dir, f'patch_{i}_{j}.tif'), single_patch)
          
          
if __name__ == "__main__":
  image_path = '/Users/clarkao1/Desktop/Hydra_Analysis/test/images/test_images_stack.tif'
  mask_path = '/Users/clarkao1/Desktop/Hydra_Analysis/test/masks/test_mask_stack.tif'
  patch_size = 256
  
  make_patches(image_path, mask_path, patch_size)