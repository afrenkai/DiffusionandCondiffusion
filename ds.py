import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


class FacesDataset(Dataset):
    def __init__(self, image_file, age_file):
        raw = np.load(image_file) # Load data from .npy file
        self.ages = np.load(age_file)
        self.data = raw / 255.
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Convert numpy array to PyTorch tensor
            transforms.RandomHorizontalFlip(),
            transforms.v2.RandomPhotometricDistort(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize data
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)
        return image, self.ages[idx]
