import torch
import numpy as np
from torch.utils.data import Dataset
from utils import load_dataset_from_mat

MAXB = 255.

class RectNetDataset(Dataset):
    def __init__(self, dataset_path:str, reorder:bool=True, xy_channels:bool=False) -> None:
        '''
        dataset_path - path of the dataset .mat file [str]
        reorder - [optional] re-ordering the four corners making top-left first to avoid ambiguity [bool] [default:True]
        xy_channels - [optional] Add XY coordinates to the input channels [default:False]
        '''
        self.images, self.corners = load_dataset_from_mat(dataset_path, 'images_raw', 'corners')

        self.length, self.height, self.width = self.images.shape

        # four corners -> absolute coordinates
        self.corners = self.corners * [self.height, self.width]

        self.reorder = reorder
        if self.reorder:
            self.topleft_idx = np.argmin(np.sum(self.corners,axis=2), axis=1)
        
        self.xy_channels = xy_channels
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image  = np.expand_dims(self.images[idx]/MAXB, axis=0)  # [H, W] -> [1, H, W] (network input B x N x H x W)
        
        if self.xy_channels:
            xy = np.mgrid[1:self.height+1, 1:self.width+1]
            image = np.concatenate([image,xy], axis=0)

        if self.reorder:
            # re-arrange the order of four corners [top-left first to avoid ambiguity]
            k = self.topleft_idx[idx]
            corner = self.corners[idx, np.arange(k,k+4)%4].flatten()      # [4, 2] -> [8]
        else:
            corner = self.corners[idx].flatten() # [4, 2] -> [8]

        return image, corner
