import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage import io

class HydraDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        #image = io.imread(image_path).astype(np.uint8)
        #mask = io.imread(mask_path).astype(np.uint8)
  
        if self.transform:
            image = self.transform(image)
            
        return [image, mask]
