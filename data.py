from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, augmentation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation
        self.images = os.listdir(self.image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = cv2.imread(image_path, 1)
        mask = cv2.imread(mask_path, 0)
  
        if self.augmentation:
            sample = {'image': image, 'mask': mask}
            sample = self.augmentation(**sample)
            image, mask = sample['image'], np.array(sample['mask'].unsqueeze(0))

        return {
            'image': image,
            'mask': mask
        }