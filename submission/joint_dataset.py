import torch
import torchvision.transforms as transforms
import torch.utils.data

from PIL import Image

import os
import random

"""
Inspired by the CycleGAN repo
Apply random augmentations to training data
Preload the images to reduce IO operations
Returns a pair of images that are randomly picked by the DataLoader
"""
class JointDomainImageDataset(torch.utils.data.Dataset):
    def __init__(self, domain_X_folder, domain_Y_folder, train, img_size):
        self.X_imgs = []
        self.Y_imgs = []

        for file_name in os.listdir(domain_X_folder):
            self.X_imgs.append(Image.open(f"{domain_X_folder}/{file_name}").convert("RGB"))
        
        for file_name in os.listdir(domain_Y_folder):
            self.Y_imgs.append(Image.open(f"{domain_Y_folder}/{file_name}").convert("RGB"))

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(1.5, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
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

"""
For training the RecycleGAN model we have to provide triplets of sequential patches
Requires a bit more work to load them in correctly
Guided by the official RecycleGAN repo
"""
class JointDomainTripletDataset(torch.utils.data.Dataset):
    def __init__(self, domain_X_folder, domain_Y_folder, train, img_size):
        self.img_size = img_size
        self.train = train

        self.X_triplets = []
        self.Y_triplets = []

        for file_name in os.listdir(domain_X_folder):
            triplet = self.__split_triplet(Image.open(f"{domain_X_folder}/{file_name}").convert("RGB"))
            self.X_triplets.append(triplet)
        
        for file_name in os.listdir(domain_Y_folder):
            triplet = self.__split_triplet(Image.open(f"{domain_Y_folder}/{file_name}").convert("RGB"))
            self.Y_triplets.append(triplet)

        self.X_size = len(self.X_triplets)
        self.Y_size = len(self.Y_triplets)

    def __split_triplet(self, triplet):
        t_0 = triplet.crop((0, 0, self.img_size, self.img_size))
        t_1 = triplet.crop((self.img_size, 0, 2 * self.img_size, self.img_size))
        t_2 = triplet.crop((2 * self.img_size, 0, 3 * self.img_size, self.img_size))

        return (t_0, t_1, t_2)

    def __getitem__(self, idx):
        X_idx = idx % self.X_size
        Y_idx = random.randint(0, self.Y_size - 1)

        x_localised_transform = []

        # Have to apply the same transform to each image in the triplet
        if random.uniform(0, 1) < 0.5 and self.train:
            x_localised_transform.append(transforms.RandomHorizontalFlip(p=1))

        if random.uniform(0, 1) < 0.25 and self.train:
            x_localised_transform.append(transforms.RandomAdjustSharpness(1.2, p=1))

        x_localised_transform += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        x_localised_transform = transforms.Compose(x_localised_transform)

        x_0, x_1, x_2 = self.X_triplets[X_idx]
        x_0 = x_localised_transform(x_0)
        x_1 = x_localised_transform(x_1)
        x_2 = x_localised_transform(x_2)

        y_localised_transform = []

        if random.uniform(0, 1) < 0.5 and self.train:
            y_localised_transform.append(transforms.RandomHorizontalFlip(p=1))

        if random.uniform(0, 1) < 0.25 and self.train:
            y_localised_transform.append(transforms.RandomAdjustSharpness(1.2, p=1))

        y_localised_transform += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        y_localised_transform = transforms.Compose(y_localised_transform)

        y_0, y_1, y_2 = self.Y_triplets[Y_idx]
        y_0 = y_localised_transform(y_0)
        y_1 = y_localised_transform(y_1)
        y_2 = y_localised_transform(y_2)

        # Return the sequential triplets
        return ((x_0, x_1, x_2), (y_0, y_1, y_2))
    
    def __len__(self):
        return max(self.X_size, self.Y_size)