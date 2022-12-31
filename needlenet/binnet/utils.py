''' Utilities '''
import cv2
import time
import h5py
import numpy as np
import scipy.io as sio

from PIL import Image
from PIL import ImageDraw

import torch
from torch import Tensor

from scipy import __version__ as scipy_version
if scipy_version >= '1.8':
    from scipy.io.matlab import matfile_version
else:
    from scipy.io.matlab.mio import _open_file
    from scipy.io.matlab.miobase import get_matfile_version

def check_matfile_version(matfile: str):
    if scipy_version >= '1.8':
        return matfile_version(matfile, appendmat=True)[0]
    else:
        return get_matfile_version(_open_file(matfile, appendmat=True)[0])[0]


def load_mat_file(matfile: str):
    '''
    Load dataset from .mat file (MATLAB '-v7.2' or below and '-v7.3')
    '''
    if check_matfile_version(matfile) < 2:
        # for '-v7.2' and lower version of .mat file (MATLAB)
        file = sio.loadmat(matfile)
    else:  # MATLAB .mat v7.3
        file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    return file


def load_dataset_from_mat(matfile: str, images_varname: str = 'images', labels_varname='labels'):
    '''
    Load dataset (image/label pairs [0]->N) from MATLAB .MAT file
    '''
    start_time = time.time()
    if check_matfile_version(matfile) < 2:
        # for '-v7.2' and lower version of .mat file (MATLAB)
        file = sio.loadmat(matfile)
        images = np.array(file[images_varname])
        labels = np.array(file[labels_varname])
    else:  # MATLAB .mat v7.3
        file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        images = np.transpose(file[images_varname])
        labels = np.transpose(file[labels_varname])
    print('dataset loading elapsed time %.2f seconds. dataset file: %s' %
          ((time.time() - start_time), matfile))
    return images, labels


class addSaltandPepperNoise(object):
    '''
    Add `Salt-and`Pepper` impulse noise to image as a data augmentation transform
    '''

    def __init__(self, density: float = 0.001, lower: float = 5./255, upper: float = 250./255):
        self.density = density
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        randmat = torch.rand(image.shape)
        image[randmat >= (1-self.density)] = self.upper
        image[randmat <= self.density] = self.lower
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(density={0}, lower={1}, upper={2})'.format(self.density, self.lower, self.upper)


def generate_mask_from_corners(shape, corners):
    '''
    Generate a binary mask from (four) corners where the inside is `1` and 
    the rest is `0`.
    '''
    blank = Image.new('1', shape, 0) # binary PIL image - all 0's
    draw = ImageDraw.Draw(blank)

    corners_draw = [tuple(corners[i]) for i in range(corners.shape[0])]
    draw.polygon(corners_draw, fill='white', outline=None)

    return np.asarray(blank).astype(np.uint8)


def draw_polygon_on_image(image, corner, colormode=False, edgecolor='blue', pointcolor='cyan'):
    if colormode:
        im_pil = Image.fromarray(np.repeat(image[..., np.newaxis], 3, axis=-1), 'RGB')
    else:
        im_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(im_pil)

    corner_draw = [tuple(corner[i]) for i in range(corner.shape[0])]
    draw.polygon(corner_draw, fill=None, outline=edgecolor)
    draw.point(tuple(corner[0]), fill=pointcolor)

    return np.asarray(im_pil)


def crop_homography(image, corner, dst_size=[100,100]):
    H, W = dst_size
    pts_dst = np.array([[0,0], [W,0], [W,H], [0,H]])

    # compute homography
    Hmat, status = cv2.findHomography(np.float32(corner), pts_dst)

    im_dst = cv2.warpPerspective(image, Hmat, [W,H])

    return im_dst


def dice_coeff(input:Tensor, target:Tensor, reduce_batch_first:bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input:Tensor, target:Tensor, reduce_batch_first:bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:,
                           channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input:Tensor, target:Tensor, multiclass:bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
