import os
import sys
import glob
import tifffile as tiff
from tqdm import tqdm
from patchify import patchify


def make_patches(image_path, mask_path):
  imgs = glob.glob(os.path.join(image_path, '*.tif'))
  mks = glob.glob(os.path.join(mask_path, '*.tif'))
  sys.stdout.flush()

  print("Processing images...")
  sys.stdout.flush()
  for img in tqdm(imgs, total=len(imgs)):
    large = tiff.imread(img)
    img_patches = patchify(large, (256, 256), step=256)
    
    for i in range(img_patches.shape[0]):
      for j in range(img_patches.shape[1]):
        single_patch = img_patches[i,j,:,:]
        imgs_dir = 'patches/images'
        os.makedirs(imgs_dir, exist_ok = True)
        tiff.imwrite(os.path.join(imgs_dir, f'patch_{i}_{j}.tif'), single_patch)
              
  print('Processing masks...')
  sys.stdout.flush()
  for msk in tqdm(mks, total=len(mks)):
      large = tiff.imread(msk)
      msk_patches = patchify(large, (256, 256), step=256)
      
      for i in range(msk_patches.shape[0]):
        for j in range(msk_patches.shape[1]):
          single_patch = msk_patches[i,j,:,:]
          single_patch = single_patch / 255.
          mks_dir = 'patches/masks'
          os.makedirs(mks_dir, exist_ok = True)
          tiff.imwrite(os.path.join(mks_dir, f'patch_{i}_{j}.tif'), single_patch)
          
          
if __name__ == "__main__":
  image_path = '/Users/clarkao1/Desktop/train/images'
  mask_path = '/Users/clarkao1/Desktop/train/masks'
  
  make_patches(image_path, mask_path)