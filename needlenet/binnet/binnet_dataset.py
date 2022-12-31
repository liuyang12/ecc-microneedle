import torch
import numpy as np
from random import uniform, randrange
from torchvision import transforms
from torch.utils.data import Dataset
from utils import (load_dataset_from_mat, 
                   addSaltandPepperNoise,
                   generate_mask_from_corners)

MAXB = 255.

class BinNetDataset(Dataset):
    def __init__(self, dataset_path:str, use_corner_mask=False, add_impulse_noise:bool=False) -> None:
        '''
        dataset_path - path of the dataset .mat file [str]
        use_corner_mask - use corners to generate binarized masks [bool] [default:False]
        add_impulse_noise - add impulse noise (implemented by random erasing of small regions) [bool] [default:False]
        '''
        self.use_corner_mask = use_corner_mask
        if self.use_corner_mask:
            self.images, self.corners = load_dataset_from_mat(dataset_path, 'images_raw', 'corners')
            # four corners -> absolute coordinates
            self.corners = self.corners * self.images.shape[1:3]
        else:
            self.images, self.masks = load_dataset_from_mat(dataset_path, 'images_raw', 'images_bin')

        self.length, self.height, self.width = self.images.shape

        self.add_impulse_noise = add_impulse_noise
        if self.add_impulse_noise:
            sz = self.height * self.width
            self.random_erase = transforms.Compose([
                transforms.ToTensor(),
                addSaltandPepperNoise(),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.RandomErasing(p=.8,scale=(.1/sz,25./sz),ratio=(0.3,3.3),value=uniform(0.6,1)),
                transforms.GaussianBlur(kernel_size=randrange(1,9,2),sigma=(0.1,3.0))
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.use_corner_mask:
            mask = generate_mask_from_corners((self.width,self.height), self.corners[idx])
        else:
            mask = self.masks[idx]
        
        if self.add_impulse_noise:
            image = self.random_erase(self.images[idx]).float().contiguous() # [H, W] -> [1, H, W]
            mask = torch.as_tensor(mask.copy()).long().contiguous() # [H, W]
        else:
            image = np.expand_dims(self.images[idx]/MAXB, axis=0) # [H, W] -> [1, H, W]
            # mask  = np.expand_dims(self.masks[idx], axis=0)  # [H, W] -> [1, H, W]
            mask = mask.astype(np.uint8)  # [H, W]

        return image, mask
