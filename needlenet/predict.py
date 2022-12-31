import os
import logging
import argparse
import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision import transforms

from model import NeedleNet
from utils.data_loading import BasicDataset
from utils.utils import (im_crop_bbox,
                         im_crop_bbox_binnet,
                         plot_img_and_mask, 
                         plot_save_image_mask,
                         get_dst_imsize_from_needle_size)

CHECKPOINT = '../checkpoints/model_10x10_100k.pth'
FILE_EXT = 'jpg'
CSV_FILE = 'needlenet_bit_error_rates.csv'
NUM_CHANNELS = 1 # [fixed] Number of color channels as the NeedleNet input
NUM_CLASSES  = 2 # [fixed] Number of classes [ON/OFF] to be predicted

NORM_MAX = True # Normalize the image with the maximum to 1

BIN_METHOD = 'thresh'  # binarization method - binnarization network
# BIN_METHOD = 'binnet' # binarization method - binnarization network
BIN_SIZE = (256, 256) # input size for binarization
CENTER_CROP = True # center crop before resizing for binarization

MAXB = 256
# intensity_threshold = 60*15*15
# intensity_threshold_factor = 0.6*15*15
# intensity_threshold_factor = 0.6*20*20
intensity_threshold_factor = 0.8*20*20
# intensity_threshold_factor = 3.5*20*20 # [temporal]
# intensity_threshold_factor = 0.98  # [temporal]
idx_threshold_min = 2
idx_threshold_max = MAXB-1
bboxmag = 1.35         # magnification factor of the bounding box
im_filter_size = 5
bin_filter_size = 5
bin_morph_size = 0


def microneedle_array_from_raw_image(im, needle_size, net=None, model=CHECKPOINT, norm_max=NORM_MAX, bin_method=BIN_METHOD, binnet=None, bin_size=BIN_SIZE, center_crop=CENTER_CROP):
    # [1] load the pretrained NeedleNet model if not exists
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if net is None:
        net = load_needlenet(model, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)

    # [2] read and pre-process the image for network evaluation
    # im = cv.imread(imfile, 0) # read image as grayscale
    intensity_threshold = needle_size*needle_size*intensity_threshold_factor
    dst_size = get_dst_imsize_from_needle_size(needle_size)
    if bin_method == 'binnet':
        im_crop = im_crop_bbox_binnet(im, binnet, bin_size=bin_size, center_crop=center_crop, bboxmag=bboxmag, dst_size=dst_size, im_filter_size=0, norm_max=norm_max)
    elif bin_method == 'thresh':
        im_crop = im_crop_bbox(im, needle_size, intensity_threshold, im_filter_size, MAXB, idx_threshold_min, idx_threshold_max, bin_filter_size, bboxmag, dst_size, bin_morph_size)
    else:
        im_crop = im_crop_bbox(im, needle_size, intensity_threshold, im_filter_size, MAXB, idx_threshold_min, idx_threshold_max, bin_filter_size, bboxmag, dst_size, bin_morph_size)
    
    # [3] evaluate NeedleNet on the input image
    mask = predict_cropped_image(net, im_crop, device, norm_max=norm_max)
    
    microneedle_array = np.argmax(mask, axis=0)
    return microneedle_array, im_crop


def load_needlenet(model, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = NeedleNet(num_channels=num_channels, num_classes=num_classes)

    net.to(device=device)

    net.load_state_dict(torch.load(model, map_location=device))

    net.eval()

    return net

def predict_cropped_image(net, im, device, norm_max=True):
    if norm_max:
        im = im / np.max(im)
    else:
        im = im / (MAXB-1)
    im = torch.from_numpy(im).unsqueeze(0).unsqueeze(0)
    im = im.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(im)

        probs = F.softmax(output, dim=1)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

        # print(full_mask.shape)
    return F.one_hot(full_mask.argmax(dim=0), net.num_classes).permute(2, 0, 1).numpy()

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                needle_size=10,
                out_threshold=0.5):
    net.eval()
    dst_size = get_dst_imsize_from_needle_size(needle_size)
    img = torch.from_numpy(BasicDataset.preprocess(
        full_img, scale_factor, is_mask=False, dst_size=dst_size))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # print(img.shape, torch.min(img), torch.max(img))
        output = net(img)
        # print(output.shape, torch.min(output), torch.max(output))

        if net.num_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

        # print(full_mask.shape)

    if net.num_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.num_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--save-fig', '-f', action='store_true',
                        help='save the figure ')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')


    parser.add_argument('--needle-size', '-d', dest='needle_size',
                    metavar='N', type=int, default=10, help='Size of the microneedle array (10x10 or 12x12 or 17x17)')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))

def add_suffix_filename(in_fn, suffix='out'):
    p = Path(in_fn)

    q = Path(p.parent, suffix)
    q.mkdir(exist_ok=True)

    out_fn = Path(q, p.stem+'_'+suffix+p.suffix)
    return out_fn

    # split = os.path.splitext(in_fn)
    # return f'{split[0]}_{suffix}{split[1]}'

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    inputs = args.input

    for kin, in_files in enumerate(inputs):
        if Path(in_files).is_dir():
            in_files = Path(in_files).glob('*.png')
        else:
            in_files = [in_files]

        # out_files = get_output_filenames(args)

        # net = UNet(num_channels=3, num_classes=2)
        net = NeedleNet(num_channels=1, num_classes=2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Loading model {args.model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(args.model, map_location=device))

        logging.info('Model loaded!')

        for i, filename in enumerate(in_files):
            logging.info(f'\nPredicting image {filename} ...')
            img = Image.open(filename)

            mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            needle_size=args.needle_size,
                            out_threshold=args.mask_threshold,
                            device=device)

            # print(mask.shape, np.min(mask), np.max(mask))

            # print(mask)

            if not args.no_save:
                # out_filename = out_files[i]
                out_filename = add_suffix_filename(filename, suffix='out')
                result = mask_to_image(mask)
                result.resize((100,100), Image.NEAREST)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

                if args.save_fig:
                    fig_filename = add_suffix_filename(filename, suffix='fig')
                    plot_save_image_mask(img, np.array(result), fig_filename)

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)
