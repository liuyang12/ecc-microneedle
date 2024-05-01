import time
import h5py
import cv2 as cv
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import torch
from binnet import predict_binarization_from_raw_image

from packaging.version import Version
from scipy import __version__ as scipy_version
if Version(scipy_version) >= Version('1.8'):
    from scipy.io.matlab import matfile_version
else:
    from scipy.io.matlab.mio import _open_file
    from scipy.io.matlab.miobase import get_matfile_version


def check_matfile_version(matfile: str):
    if Version(scipy_version) >= Version('1.8'):
        return matfile_version(matfile, appendmat=True)[0]
    else:
        return get_matfile_version(_open_file(matfile, appendmat=True)[0])[0]

def load_mat_file(matfile:str):
    '''
    Load dataset from .mat file (MATLAB '-v7.2' or below and '-v7.3')
    '''
    if check_matfile_version(matfile) < 2:
        # for '-v7.2' and lower version of .mat file (MATLAB)
        file = sio.loadmat(matfile)
    else:  # MATLAB .mat v7.3
        file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    return file


def load_dataset_from_mat(matfile:str, images_varname:str='images', labels_varname='labels'):
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


def get_dst_imsize_from_needle_size(needle_size):
    # if needle_size == 10:
    #     dst_size = [120, 120]
    # elif needle_size == 12:
    #     dst_size = [136, 136]
    # else:
    #     print(f'Unsupported needle size {needle_size}, using default dst_size (120,120).')
    #     dst_size = [120,120]
    side_size = ((needle_size + 4) * 2 + 2) * 4

    dst_size = [side_size, side_size]

    return dst_size


'''crop image for recognition from binarized raw image'''


def im_crop_bbox_binnet(im, binnet=None, device=None, bin_size=(256, 256), center_crop=False, im_filter_size=0, bboxmag=1.0, dst_size=(360, 360), norm_max=True, MAXB=256.):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = im.shape
    if center_crop and (H!=W):
        a = min(H,W)
        xc = (W - a) // 2
        yc = (H - a) // 2
        im = im[yc:yc+a, xc:xc+a]
    if im.shape != bin_size:
        im = cv.resize(im, bin_size)
    if im_filter_size > 0: # median filter to remove impulse noise (where BinNet is super sensitive)
        im_filter = cv.medianBlur(im, im_filter_size)
    else:
        im_filter = im
    # print(im.shape)
    im_bin = predict_binarization_from_raw_image(im_filter, binnet, device=device, norm_max=norm_max)

    # im_bin = np.uint8(im_bin*(MAXB-1))
    # print(im_bin.shape, np.max(im_bin), np.min(im_bin))
    # if bin_morph_size > 0:
    #     # apply morphology
    #     morph_kernel = cv.getStructuringElement(
    #         cv.MORPH_ELLIPSE, (bin_morph_size, bin_morph_size))
    #     im_bin = cv.morphologyEx(im_bin, cv.MORPH_OPEN, morph_kernel)
    
    coords = np.column_stack(np.where(im_bin > 0.5))[:, [1, 0]]

    rect = cv.minAreaRect(coords)
    size_new = tuple(l*bboxmag for l in rect[1])
    rect = (rect[0], size_new, rect[2]) # center, size, angle

    im_crop, _ = crop_rect(im, rect)
    if im_crop is not None:
        im_crop = cv.resize(cv.rotate(im_crop, 2), dst_size)

    # im_crop_bin, _ = crop_rect(im_bin, rect)
    # if im_crop_bin is not None:
    #     im_crop_bin = cv.resize(cv.rotate(im_crop_bin, 2), dst_size)

    return im_crop


def im_crop_bbox(im, needle_size, intensity_threshold, im_filter_size=0, MAXB=256, idx_threshold_min=2, idx_threshold_max=255, bin_filter_size=0, bboxmag=1.0, dst_size=(360, 360), bin_morph_size=0, crop_boundary=False, outlier_removal=False):
    if crop_boundary:
        r = 8 # (1-2/r)
        [H, W] = im.shape
        im = im[H//r:-H//r,W//r:-W//r]
    if im_filter_size > 0:
        im_filter = cv.medianBlur(im, im_filter_size)
    else:
        im_filter = im.copy()
    hist, bin_edges = np.histogram(im_filter.ravel(), MAXB, [0, MAXB])

    revsum = np.cumsum(hist[::-1])

    if intensity_threshold/(needle_size*needle_size) > 1:
        # hard global threshold based on absolute value of intensity  histogram
        idx_threshold = MAXB - np.argwhere(revsum > intensity_threshold)[0] - 1
    else:
        # hard global threshold based on relative value of intensity  histogram
        intensity_threshold = intensity_threshold/(needle_size*needle_size)
        idx_threshold = MAXB - \
            np.argwhere(revsum > intensity_threshold*revsum[-1])[0] - 1
    idx_threshold = max(idx_threshold, idx_threshold_min)
    idx_threshold = min(idx_threshold, idx_threshold_max)

    im_bin = np.uint8((im_filter >= idx_threshold)*(MAXB-1))

    if bin_filter_size > 0:
        im_bin = cv.medianBlur(np.uint8(im_bin), bin_filter_size)
    
    if bin_morph_size > 0:
        # apply morphology
        morph_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (bin_morph_size, bin_morph_size))
        im_bin = cv.morphologyEx(im_bin, cv.MORPH_OPEN, morph_kernel)

    if outlier_removal:
        inlier_segmap1 = covariance_fitting(im_bin, 2)
        inlier_segmap = covariance_fitting(im_bin*inlier_segmap1, 2.5)
        im_bin = im_bin*inlier_segmap

    coords = np.column_stack(np.where(im_bin > MAXB/2))[:, [1, 0]]

    im_box = im.copy()
    rect = cv.minAreaRect(coords)
    size_new = tuple(l*bboxmag for l in rect[1])
    rect = (rect[0], size_new, rect[2])

    im_crop, _ = crop_rect(im, rect)
    if im_crop is not None:
        im_crop = cv.resize(cv.rotate(im_crop, 2), dst_size)

    # im_crop_bin, _ = crop_rect(im_bin, rect)
    # if im_crop_bin is not None:
    #     im_crop_bin = cv.resize(cv.rotate(im_crop_bin, 2), dst_size)

    return im_crop


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1)
    # rotate the original image 
    # #   zero padding for affine transform boundaries
    img_rot = cv.warpAffine(img, M, (width, height),
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=0)
    #   mirror padding for affine transform boundaries
    # img_rot = cv.warpAffine(img, M, (width, height),
    #                         borderMode=cv.BORDER_REPLICATE)

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def covariance_fitting(im, variance_range=2):
    [H, W] = im.shape
    xx, yy = np.meshgrid(np.linspace(0, W, W), np.linspace(0, H, H))

    X = np.stack(np.where(im > 0), axis=0)
    cov_mat = np.cov(X)

    sigma_inv = fractional_matrix_power(cov_mat, -0.5)

    z0 = np.average(X, axis=1)
    z = np.stack([yy.flatten(), xx.flatten()], axis=1)
    dz = z - z0
    z_norm = dz @ sigma_inv

    inlier_seg = np.amax(np.abs(z_norm), axis=1) <= variance_range

    inlier_segmap = inlier_seg.reshape([H, W])

    return inlier_segmap


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_save_image_mask(im, mask, filename):
    plt.ioff()
    fig = plt.figure(figsize=(8,5),dpi=150)
    plt.subplot(121)
    plt.imshow(im, cmap='gray')
    plt.title('Input image')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.title('Predicted binary array')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def add_suffix_filename(in_fn, suffix='out', out_ext='.png'):
    p = Path(in_fn)

    q = Path(p.parent, suffix)
    q.mkdir(exist_ok=True)

    out_fn = Path(q, p.stem+'_'+suffix+out_ext)
    return out_fn
