import torch
import torchvision.transforms as transforms
import torch.utils.data

from PIL import Image

import os
import random

class JointDomainImageDataset(torch.utils.data.Dataset):
    def __init__(self, domain_X_folder, domain_Y_folder, train):
        self.X_imgs = []
        self.Y_imgs = []

        for file_name in os.listdir(domain_X_folder):
            self.X_imgs.append(Image.open(f"{domain_X_folder}/{file_name}").convert("RGB"))
        
        for file_name in os.listdir(domain_Y_folder):
            self.Y_imgs.append(Image.open(f"{domain_Y_folder}/{file_name}").convert("RGB"))

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(286, transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(286, transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.X_size = len(self.X_imgs)
        self.Y_size = len(self.Y_imgs)

    def __getitem__(self, idx):
        X_idx = idx % self.X_size
        Y_idx = random.randint(0, self.Y_size - 1)

        X_item = self.transform(self.X_imgs[X_idx])
        Y_item = self.transform(self.Y_imgs[Y_idx])

        return (X_item, Y_item)
    
    def __len__(self):
        return max(self.X_size, self.Y_size)