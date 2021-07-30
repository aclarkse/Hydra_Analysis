import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage import io

class HydraDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
                        transforms.Resize((1600, 2084)),
                        transforms.ToTensor()])
        self.images = sorted(Path(image_dir).glob('*.tif'))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images[index])
        mask_path = os.path.join(self.images[index])
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
  
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return [image, mask]
