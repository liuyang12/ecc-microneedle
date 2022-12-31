''' Utilities '''
import cv2
import time
import h5py
import numpy as np
import scipy.io as sio

from PIL import Image
from PIL import ImageDraw

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

