import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import (get_dst_imsize_from_needle_size, load_dataset_from_mat)

class BasicDataset(Dataset):
    def __init__(self, dataset_path: str, scale: float = 1.0, mask_suffix: str = '', needle_size: int = 10, single_dataset_file: bool = None):
        if single_dataset_file is not None:
            self.single_dataset_file = single_dataset_file
            if single_dataset_file:
                self.images, self.masks = load_dataset_from_mat(dataset_path, 'images_rect', 'labels')
            else:
                self.images_dir = Path(dataset_path + '/image/')
                self.masks_dir  = Path(dataset_path + '/mask/')
        else:
            dpath = Path(dataset_path)
            if dpath.is_file():
                self.single_dataset_file = True
                self.images, self.masks = load_dataset_from_mat(dataset_path, 'images_rect', 'labels')
            elif dpath.is_dir():
                self.single_dataset_file = False
                self.images_dir = Path(dataset_path + '/image/')
                self.masks_dir  = Path(dataset_path + '/mask/')
            else:
                raise RuntimeError(f'No dataset file or directory found in {self.dataset_path}.')
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.needle_size = needle_size
        self.dst_size = get_dst_imsize_from_needle_size(needle_size)
        self.resize = (scale!=1) and (self.images.shape[1:]!=self.dst_size)

        if self.single_dataset_file:
            self.length = self.images.shape[0]
        else:
            self.ids = [splitext(file)[0] for file in listdir(
                self.images_dir) if not file.startswith('.')]
            if not self.ids:
                raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
            logging.info(f'Creating dataset with {len(self.ids)} examples')

            self.length = len(self.ids)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.single_dataset_file:
            img  = self.images[idx]
            mask = self.masks[idx]
        else:
            name = self.ids[idx]
            mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            mask = self.load(mask_file[0])
            img = self.load(img_file[0])

        img  = self.preprocess(img,  self.scale, is_mask=False, dst_size=self.dst_size, resize=self.resize)
        mask = self.preprocess(mask, self.scale, is_mask=True, resize = self.resize)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask' : torch.as_tensor(mask.copy()).long().contiguous()
        }


    @classmethod
    def preprocess(cls, pil_img, scale, is_mask, dst_size=None, resize=False):
        if resize:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH))
            if dst_size is not None:
                pil_img = pil_img.resize(dst_size)
                
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if is_mask:
            img_ndarray = img_ndarray.astype(np.uint8)
        else:
            img_ndarray = img_ndarray / 255
            # img_ndarray = img_ndarray / np.max(img_ndarray)

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

class CarvanaDataset(BasicDataset):
    def __init__(self, dataset_path, scale=1):
        super().__init__(dataset_path, scale, mask_suffix='_mask')


class NeedleDataset(BasicDataset):
    def __init__(self, dataset_path, scale=1, needle_size=10):
        super().__init__(dataset_path, scale, mask_suffix='_mask', needle_size=needle_size)
