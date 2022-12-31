
# %%
# import external packages
import csv
import time
import glob
import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# import internal packages
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from predict import (load_needlenet, microneedle_array_from_raw_image)
from binnet import load_binnet
from utils.utils import add_suffix_filename

NEEDLE_SIZE = 10
NUM_TRAINING = 650 # k
IMAGE_DIR = '/data/yliu/docs/Dropbox(MIT)/Vaccine_Tracking2/Real Images/Real Images/96-bit MNP/'
FILE_EXT = 'jpg'
CSV_FILE = 'needlenet_bit_error_rates.csv'
SAVE_FIG = False
FIG_SUBDIR = 'fig' # name of subdirectory to save figures
IMAGE_MODE = False # image mode: save individual BER for each image
                   # folder mode: save the smallest BER from each subfolder
ERROR_AFTER_TRANSFORM = True # error bits before orientation-correcting transform - flipping and rotation
BIN_METHOD = 'binnet'  # binarization method - 'binnet'
BINNET_NUM_TRAINING = 900  # BinNet number of training images (k)
BINNET_SUFFIX = '_impulse'

plt.ioff()

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (1.0, 0.8, 0.8)),

          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue':  ((0.0, 0.8, 0.8),
                    (0.5, 0.0, 0.0),
                    (1.0, 0.0, 0.0))
          }
blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)


def get_per_image_bit_error_rate(imfile, needle_size, needlenet, save_fig=SAVE_FIG, fig_subdir=FIG_SUBDIR, error_after_transform=ERROR_AFTER_TRANSFORM, bin_method=None, binnet=None):

    # predict binary microneedle array from the raw input image
    im = cv.imread(imfile, 0)
    needle, im_crop = microneedle_array_from_raw_image(im, needle_size, net=needlenet, bin_method=bin_method, binnet=binnet)

    # %% [markdown]
    # ## Calculating the bit error rate (BER) according to the template
    # Note that we use the four corners for orientation detection and then get the right orientation of the template and finally include four corners in the bir error rate (BER) as well.

    # %%
    # [0] layout of the microneedle array
    #     only use 64 bits for encoding RM(1,8) or [64,7,32]_2-code
    s = needle_size   # side of the microneedle array - s x s
    p =  2   # preserved 4 corners for orientation (p x p top-right)

    # [0.1] show a blank microneedle array with a template
    blank = np.ones((s,s), dtype=int) # blank array with all zeros (binary 0/1 )
    blank[0:p,0:p] = 1;   # top-left
    blank[-p:,0:p] = 1;   # bottom-left
    blank[0:p,-p:] = 0;   # top-right
    blank[-p:,-p:] = 1;   # bottom-right 

    # [1] flip the binary pixel array for deployment
    noisyneedle = np.fliplr(needle).astype(int)

    # orientation detection (based on the black corner)
    c1 = np.sum(noisyneedle[0:p, 0:p])   # top-left
    c2 = np.sum(noisyneedle[0:p, -p:])   # top-right
    c3 = np.sum(noisyneedle[-p:, -p:])
    c4 = np.sum(noisyneedle[-p:, 0:p])   # bottom-left

    rotnum = np.argmin([c1,c2,c3,c4]) + 3

    binneedle = np.rot90(noisyneedle, rotnum)

    if error_after_transform:
        errorneedle = binneedle-blank
    else:
        errorneedle = needle.astype(int) - np.fliplr(np.rot90(blank.astype(int), -rotnum))

    num_error_bits = np.count_nonzero(errorneedle) # number of flipped pixel bits

    ber = num_error_bits/(s*s) # bit error rate

    print('  Number of error pixels: %d' % num_error_bits)
    print('  Bit error rate: {:.2%}'.format(ber))
    
    if save_fig:
        fig=plt.figure(figsize=(10, 14), dpi=300)
        plt.subplot(322)
        plt.imshow(im, cmap='gray')
        plt.title('Raw input image (grayscale)')
        plt.subplot(321)
        plt.imshow(im_crop, cmap='gray')
        plt.title('Cropped input image')
        plt.subplot(323)
        plt.imshow(needle, cmap='gray')
        plt.title('Recognized (before correcting orientation)')
        plt.subplot(324)
        plt.imshow(binneedle, cmap='gray')
        plt.title('Recognized (after correcting orientation)')
        plt.subplot(325)
        plt.imshow(errorneedle, cmap=blue_red1)
        plt.clim(-1, 1)
        plt.title('Error bit rate {:.2%} (white - error; black - correct)'.format(ber))
        plt.subplot(326)
        plt.imshow(blank, cmap='gray')
        plt.title('Full blank microneedle array (original)')
        plt.savefig(add_suffix_filename(imfile, fig_subdir), bbox_inches='tight')
        plt.close(fig)

    return ber, num_error_bits


def get_bit_error_rate(imdir, needle_size, needlenet, csvfilename, num_training=NUM_TRAINING, file_ext=FILE_EXT, save_fig=SAVE_FIG, fig_subdir=FIG_SUBDIR, image_mode=IMAGE_MODE, error_after_transform=ERROR_AFTER_TRANSFORM, bin_method=None, binnet=None):
    csvext = csvfilename.split('.')[-1]
    binmeth_suffix = '_binnet' if binnet else ''
    if image_mode:
        csvfilename = csvfilename[:-len(csvext)-1] + binmeth_suffix + '_perimage_'  + f'{num_training}k.' + csvext
        # csv fields
        fields = ['File name','Number of error bits','Bit error rate','Full file path']
    else:
        csvfilename = csvfilename[:-len(csvext)-1] + binmeth_suffix + '_perinstance_'  + f'{num_training}k.' + csvext
        # csv fields
        fields = ['File name','Minimum number of error bits','Minimum bit error rate','Full file path']

        last_instancepath = None
        last_minfname = None
        last_minneb = -1
        last_minber = -1
        
    # writing to csv file 
    with open(csvfilename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fields)

        subdirlist = natsorted(glob.glob(imdir, recursive=True))
        print(len(subdirlist))

        for subdir in subdirlist:
            imfilelist = natsorted(glob.glob(subdir+'/*.'+file_ext))
            for imfile in imfilelist:
                fstem = Path(imfile).stem
                fname = Path(imfile).name

                print(imfile)
                
                instancepath = imfile[:-len(imfile.split("_")[-1])-1]

                if not image_mode and instancepath != last_instancepath:
                    if last_instancepath is None: # first-ever image
                        last_instancepath = instancepath
                    else:
                        # record last instance
                        if last_minber >= 0: # update last instance w/ any recognizable images
                            file_row = [last_minfname, '{}'.format(last_minneb), '{:.2%}'.format(last_minber), last_instancepath]
                        else: # update last instance w/ no recognizable images
                            file_row = [last_minfname, 'FAILED', 'FAILED', last_instancepath]
                        csvwriter.writerow(file_row)

                        last_instancepath = instancepath
                        last_minfname = None
                        last_minneb = -1
                        last_minber = -1

                # %%
                try:
                    ber, num_error_bits = get_per_image_bit_error_rate(imfile, needle_size, needlenet, save_fig=save_fig, fig_subdir=fig_subdir, error_after_transform=error_after_transform, bin_method=bin_method, binnet=binnet)

                    if image_mode:
                        file_row = [fname, '{}'.format(num_error_bits), '{:.2%}'.format(ber), imfile]
                        csvwriter.writerow(file_row)
                    else:
                        if last_minneb < 0 or last_minneb > num_error_bits: # update minimum bit error rate
                            last_minfname = fname
                            last_minneb = num_error_bits
                            last_minber = ber

                except:
                    print('    Failed to get error bits from %s.' % imfile)

                    if image_mode:
                        file_row = [fname, 'FAILED', 'FAILED', imfile]
                        csvwriter.writerow(file_row)
        if subdirlist and not image_mode: # not empty list
            # record last-ever instance
            if last_minber >= 0:  # update last instance w/ any recognizable images
                file_row = [last_minfname, '{}'.format(last_minneb), '{:.2%}'.format(last_minber), last_instancepath]
            else:  # update last instance w/ no recognizable images
                file_row = [last_minfname, 'FAILED', 'FAILED', last_instancepath]
            csvwriter.writerow(file_row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--needle_size", "-a", type=int, default=NEEDLE_SIZE)
    parser.add_argument("--num_training", "-n", type=int, default=NUM_TRAINING)
    parser.add_argument("--dir", "-d", type=str, default=IMAGE_DIR)
    parser.add_argument("--file_ext", "-e", type=str, default=FILE_EXT)
    parser.add_argument("--csv_file", "-c", type=str, default=CSV_FILE)
    parser.add_argument("--save_fig", "-f", action='store_true', default=False)
    parser.add_argument("--fig_subdir", "-s", type=str, default=FIG_SUBDIR)
    parser.add_argument("--image_mode", "-i", action='store_true', default=False)
    parser.add_argument("--error_after_transform", "-t",
                        action='store_true', default=False)
    parser.add_argument("--bin_method", "-b",
                        type=str, default=BIN_METHOD)
    parser.add_argument("--binnet_num_training", "-m",
                        type=int, default=BINNET_NUM_TRAINING)
    parser.add_argument("--binnet_suffix", type=str, default=BINNET_SUFFIX)
    opts = parser.parse_args()
    
    # dirfile = os.path.abspath(opts.dir)+'/*.'+opts.file_ext
    # dirfile = opts.dir+'/**/*.'+opts.file_ext # all files in the folder [recursively]
    dirfile = opts.dir+'/**/' # all lowest-level subdirectories [recursively]

    num_training = 100  # number of training samples *1k
    model = f'../checkpoints/model_{opts.needle_size}x{opts.needle_size}_{opts.num_training}k.pth'

    # load pre-trained NeedleNet [only once]
    needlenet = load_needlenet(model)
    print(f'loaded NeedleNet {model}')

    if opts.bin_method == 'binnet':
        binnet_ckpt = f'../binnet/checkpoints/binnet{opts.binnet_suffix}_{opts.binnet_num_training}k.pth'
        binnet = load_binnet(binnet_ckpt)
        print(f'loaded BinNet {binnet_ckpt}')
    elif opts.bin_method == 'thresh':
        binnet = None
    else:
        print(f'Unknown binarization method {opts.bin_method}, using a simple threshold-based solution.')
        opts.bin_method == 'thresh'
        binnet = None

    # runtime excluding loading the network
    start = time.time()
    get_bit_error_rate(dirfile,
                       opts.needle_size,
                       needlenet, 
                       opts.dir+'/'+opts.csv_file,
                       num_training=opts.num_training,
                       file_ext=opts.file_ext, 
                       save_fig=opts.save_fig, 
                       fig_subdir=opts.fig_subdir,
                       image_mode=opts.image_mode,
                       error_after_transform=opts.error_after_transform,
                       bin_method=opts.bin_method,
                       binnet=binnet)
    end = time.time()
    print(f"\n\nTotal runtime (excluding loading NeedleNet) is {end - start} seconds\n")
    
if __name__ == "__main__":
    main()
