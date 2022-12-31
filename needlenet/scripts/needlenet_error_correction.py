
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
from reedmuller import reedmuller  # Reed-Muller code
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
IMAGE_DIR = '/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli\'s/10x10 Pattern/Applicator 3/'
FILE_EXT = 'jpg'
CSV_FILE = 'needlenet_ecc_bit_error_rates.csv'
SAVE_FIG = False
KNOWN_PATT_INFO = True # known pattern information from the filename '_PatternXX_'
FIG_SUBDIR = 'fig' # name of subdirectory to save figures
PATT_STR = '_Pattern'  # substring in the filename
# PATT_STR = '.Pattern'  # substring in the filename
IMAGE_MODE = False # image mode: save individual BER for each image
                   # folder mode: save the smallest BER from each subfolder
RM_ORDER = 2 # default RM order
ERROR_AFTER_TRANSFORM = False # error bits before orientation-correcting transform - flipping and rotation
NO_FLIP = False # no horizontal flip
BIN_METHOD = 'binnet' # binarization method - 'binnet'
BINNET_NUM_TRAINING = 900 # BinNet number of training images (k)
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

def get_per_image_ecc_bit_error_rate(imfile, needle_size, needlenet, rm_order=None, mask=None, save_fig=SAVE_FIG, fig_subdir=FIG_SUBDIR, known_patt_info=KNOWN_PATT_INFO, patt_str=PATT_STR, error_after_transform=ERROR_AFTER_TRANSFORM, no_flip=NO_FLIP, bin_method=None, binnet=None):
    # predict binary microneedle array from the raw input image
    im = cv.imread(imfile, 0)

    needle, im_crop = microneedle_array_from_raw_image(im, needle_size, net=needlenet, bin_method=bin_method, binnet=binnet)

    # [0] layout of the microneedle array
    #     only use 64 bits for encoding RM(1,8) or [64,7,32]_2-code
    if needle_size == 10:
        corner_size = 3
        corner_hole = True
        allow_inner = False
        rm_order = 1
        use_mask = False
    elif needle_size == 12:
        corner_size = 2
        corner_hole = False
        allow_inner = False
        rm_order = 2
        use_mask = True
    elif needle_size == 17:
        corner_size = 3
        corner_hole = True
        allow_inner = True
        if rm_order is None:
            rm_order = 2
        use_mask = True
    else:
        corner_size = 2
        corner_hole = False
        allow_inner = False
        if rm_order is None:
            rm_order = 1
        use_mask = False

    s = needle_size   # side of the microneedle array - s x s
    p = corner_size   # preserved 4 corners for orientation (p x p top-right)

    # [0.1] show a blank microneedle array with a template
    # blank array with all zeros (binary 0/1 )
    blank = np.zeros((s, s), dtype=bool)
    blank[0:p, 0:p] = 1   # top-left
    blank[-p:, 0:p] = 1   # bottom-left
    blank[0:p, -p:] = 1   # top-right
    blank[-p:, -p:] = 0   # bottom-right

    if corner_hole:
        blank[p//2, p//2] = 0
        blank[-p//2, p//2] = 0
        blank[p//2, -p//2] = 0

    if allow_inner:
        blank[p-1, p-1] = 0
        blank[-p, p-1] = 0
        blank[p-1,  -p] = 0

    # [2] Reed-Muller encoder
    totalbits = s*s - 4*p*p + allow_inner*3   # total number of bits for encoding

    # [2.1] RM(r,m) code
    r = rm_order                      # order of RM(r,m) code
    m = int(np.log2(totalbits))    # exponent of RM(r,m) code
    rmcode = reedmuller.ReedMuller(r, m)  # Reed-Muller RM(r,m) code
    # first-order RM(1,m) code -- [n,k,d] or [2^m, m+1, 2^(m-r)]_2-code with 2^(m-r-1)-1 maximum error correction bits
    n = rmcode.block_length()   # block length 2^m
    k = rmcode.message_length()  # message length m+1 (first-order)
    ec = rmcode.strength()      # number of correctable error bits 2^(m-r-1)-1

    # print('            Bits used for encoding:  %2d bits' % n)
    # print('                      Message bits:  %2d bits' % k)
    # print('Maximum number of correctable bits:  %2d bits' % ec)

    # [3.1] get the indices for encoding locations
    idxmat = np.arange(s*s).reshape(s, s)  # all indices for each bit
    validmat = np.ones([s, s])

    validmat[0:p, 0:p] = 0   # top-left
    validmat[-p:, 0:p] = 0   # bottom-left
    validmat[0:p, -p:] = 0   # top-right
    validmat[-p:, -p:] = 0   # bottom-right

    if allow_inner:
        validmat[p-1, p-1] = 1
        validmat[-p, p-1] = 1
        validmat[p-1,  -p] = 1

    idxvec = idxmat[validmat > 0]

    if no_flip:
        noisyneedle = needle.astype(int)
    else:
        # flip the binary
        noisyneedle = np.fliplr(needle).astype(int)

    c1 = np.sum(noisyneedle[0:p, 0:p])   # top-left
    c2 = np.sum(noisyneedle[0:p, -p:])   # top-right
    c3 = np.sum(noisyneedle[-p:, -p:])
    c4 = np.sum(noisyneedle[-p:, 0:p])   # bottom-left

    rotnum = np.argmin([c1, c2, c3, c4]) + 2

    noisyneedle = np.rot90(noisyneedle, rotnum)

    # [2.2] translate the text message (vaccine type and date ) to binary bits
    if needle_size == 10:
        # message length k = 7
        nbits = [1, 2, 4]  # number of bits for each block
        # [1-bit] type 1-2 (start from 1)
        # [2-bit] year 1-4 (start from 1)
        # [4-bit] month 1-12 (start from 1)
    elif needle_size == 12:
        nbits = [4, 4, 14, 3, 4]  # number of bits for each block
        # [4-bit] type 1-16 (start from 1)
        # [4-bit] manufacturer 1-16 (start from 1)
        # [14-bit] LOT/batch 1-16384 (start from 1)
        # [3-bit] year 1-8 (start from 1)
        # [4-bit] month 1-12 (start from 1)
    elif needle_size == 17:
        if rm_order == 2:
            # 37-bit
            nbits = [6,6,18,3,4] # number of bits for each block
            # [6-bit] type 1-64 (start from 1)
            # [6-bit] manufacturer 1-64 (start from 1)
            # [18-bit] LOT/batch 1-262144 (start from 1)
            # [3-bit] year 1-8 (start from 1)
            # [4-bit] month 1-12 (start from 1)
        elif rm_order == 1:
            # 9-bit
            nbits = [1,2,2,4] # number of bits for each block
            # [1-bit] type 1-2 (start from 1)
            # [2-bit] manufacturer 1-4 (start from 1)
            # [2-bit] year 1-4 (start from 1)
            # [4-bit] month 1-12 (start from 1)
        else:
            print('RM order %d not supported' % (rm_order))
    else:
        print('Specify number of bits for each block')

    noisyneedle_mask = noisyneedle
    if use_mask:
        noisyneedle = np.logical_xor(noisyneedle, mask)

    noisycode = np.zeros(n)
    for i in range(n):
        noisycode[i] = noisyneedle[idxvec[i]//s, idxvec[i] % s]

    # [5] run Reed-Muller error correction decoding to get the formated message
    y = rmcode.decode(noisycode)

    if y is None:
        success = False
        text_dec = ''
        print('  Reed-Muller decoding failed.')
    else:
        space = ''
        print('  Binary corrected bits:  %s' % space.join(map(str, y)))

        cumbits = np.cumsum(nbits)
        cumbits = np.append([0], cumbits)
        vinfo_dec = np.zeros([len(nbits), 1])
        text_dec = ''
        # decoded vaccine types and dates
        for ibit in range(len(nbits)):
            vinfo_dec[ibit] = int(space.join(
                map(str, y[cumbits[ibit]:cumbits[ibit+1]])), 2) + 1
            text_dec = text_dec + '%02d-' % vinfo_dec[ibit]

        text_dec = text_dec[:-1]
        print('  Decoded info: Vaccine %s' % (text_dec))

    # get ground truth pattern information from filename '_PatternXX_'
    success = True
    patt_info = 'NA'
    # [TODO] fix a bug when filename has patt_str without a number at the end
    if known_patt_info:
        # load the pattern information csv file as a dictionary for queries
        with open('../checkpoints/pattern_info.csv', mode='r') as csvfile:
            reader = csv.reader(csvfile)
            patt_info_dict = {rows[0]: rows[1] for rows in reader}

        i = 0
        while i < len(imfile):
            digit_idx = imfile.find(patt_str, i)
            if digit_idx > 0:
                if imfile[digit_idx+len(patt_str)].isdigit():
                    i = len(imfile)
                    if imfile[digit_idx+len(patt_str)+1].isdigit():
                        patt_num = imfile[digit_idx+len(patt_str):digit_idx+len(patt_str)+2]
                    else:
                        patt_num = imfile[digit_idx+len(patt_str)]

                    patt_info = patt_info_dict[patt_num]

                    info = list(map(int, patt_info.split('-')))

                    v = []
                    for ibit in range(len(nbits)):
                        binfo = list(map(int, '{0:0b}'.format(info[ibit]-1)))
                        v = v + [0]*(nbits[ibit]-len(binfo)) + binfo

                    success = (v == y)
                else:
                    i = digit_idx+len(patt_str)
            else:
                i = len(imfile)
                known_patt_info = False

    # [3.2] assign value to each bit in the microneedle
    origneedle = blank.copy()

    # blank array with all zeros (binary 0/1 )
    infomat = np.ones((s, s), dtype=bool)
    infomat[0:p, 0:p] = 0   # top-left
    infomat[-p:, 0:p] = 0   # bottom-left
    infomat[0:p, -p:] = 0   # top-right
    infomat[-p:, -p:] = 0   # bottom-right

    # needle[idxvec] = code
    if known_patt_info:
        origcode = rmcode.encode(v)
    else:
        origcode = rmcode.encode(y)
    for i in range(n):
        origneedle[idxvec[i]//s, idxvec[i] % s] = origcode[i]

    if use_mask:
        origneedle = np.logical_xor(origneedle, mask).astype(int)

    if error_after_transform:
        errorneedle = noisyneedle_mask-origneedle.astype(int)
    elif no_flip:
        errorneedle = needle.astype(int) - np.rot90(origneedle.astype(int), -rotnum)
    else:
        errorneedle = needle.astype(int) - np.fliplr(np.rot90(origneedle.astype(int), -rotnum))

    num_error_bits = np.count_nonzero(errorneedle)  # number of flipped pixel bits
    num_info_error_bits = np.count_nonzero(
        errorneedle*infomat)  # number of flipped pixel bits

    ber = num_error_bits/(s*s)  # bit error rate
    if not known_patt_info:
        success = (num_info_error_bits <= ec)

    print('  Number of error pixels: %d' % num_error_bits)
    print('  Bit error rate: {:.2%}'.format(ber))
    print('  Number of info error pixels (excluding four %dx%d corners): %d' %
        (s, s, num_info_error_bits))
    if success:
        print('  Successful error correction!')
    else:
        print('  Error correction failed.')

    if save_fig:
        fig = plt.figure(figsize=(10, 14), dpi=300)
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
        plt.imshow(noisyneedle_mask, cmap='gray')
        plt.title('Recognized (after correcting orientation)')
        plt.subplot(326)
        plt.imshow(origneedle, cmap='gray')
        plt.title('Original microneedle array [after masking]')
        plt.subplot(325)
        plt.imshow(errorneedle, cmap=blue_red1)
        plt.clim(-1,1)
        plt.title('Error bit rate {:.2%} (white - error; black - correct)'.format(ber))
        plt.savefig(add_suffix_filename(imfile, fig_subdir), bbox_inches='tight')
        plt.close(fig)

    return ber, num_error_bits, success, num_info_error_bits, text_dec, patt_info


def get_ecc_bit_error_rate(imdir, needle_size, needlenet, csvfilename, num_training=NUM_TRAINING, file_ext=FILE_EXT, save_fig=SAVE_FIG, fig_subdir=FIG_SUBDIR, known_patt_info=KNOWN_PATT_INFO, patt_str=PATT_STR, image_mode=IMAGE_MODE, rm_order=RM_ORDER, error_after_transform=ERROR_AFTER_TRANSFORM, no_flip=NO_FLIP, bin_method=None, binnet=None):
    # load mask if using a mask for encoding
    mask = None
    if needle_size > 10:
        import scipy.io as sio
        maskfile = f'../checkpoints/mask_{needle_size}x{needle_size}.mat'
        if os.path.isfile(maskfile):
            data = sio.loadmat(maskfile)
            mask = data['mask']
            print(f'  loaded mask file {maskfile}')
        else:
            mask = None
            print(f'  mask file {maskfile} not found')
    csvext = csvfilename.split('.')[-1]
    binmeth_suffix = '_binnet' if binnet else ''
    if image_mode:
        csvfilename = csvfilename[:-len(csvext)-1] + binmeth_suffix + '_perimage_'  + f'{num_training}k.' + csvext
        # csv fields
        fields = ['File name','Original info','Decoded info','Number of error bits','Bit error rate','Number of info error bits (excluding four corners)','Full file path']
    else:
        csvfilename = csvfilename[:-len(csvext)-1] + binmeth_suffix + '_perinstance_'  + f'{num_training}k.' + csvext
        # csv fields
        fields = ['File name','Original info','Decoded info','Minimum number of error bits','Minimum bit error rate','Minimum number of info error bits (excluding four corners)','Full file path']

        last_instancepath = None
        last_minfname = None
        last_originfo = None
        last_decinfo = None
        last_success = False
        last_minneb = -1
        last_minber = -1
        last_mininfoneb = -1
        
    # writing to csv file 
    with open(csvfilename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fields)

        subdirlist = natsorted(glob.glob(imdir,recursive=True))
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
                        if last_success: # update successful recognization
                            file_row = [last_minfname, last_originfo, 'SUCCESS: '+last_decinfo, '{}'.format(last_minneb), '{:.2%}'.format(last_minber), '{}'.format(last_mininfoneb), last_instancepath]
                        elif last_minber >= 0: # update last instance w/ any recognizable images
                            file_row = [last_minfname, last_originfo, 'FAILED: '+last_decinfo, '{}'.format(last_minneb), '{:.2%}'.format(last_minber), '{}'.format(last_mininfoneb), last_instancepath]
                        else: # update last instance w/ no recognizable images
                            file_row = [last_minfname, last_originfo, 'FAILED', 'FAILED', 'FAILED', 'FAILED', last_instancepath]
                        csvwriter.writerow(file_row)

                        last_instancepath = instancepath
                        last_minfname = None
                        last_originfo = None
                        last_decinfo = None
                        last_success = False
                        last_minneb = -1
                        last_minber = -1
                        last_mininfoneb = -1
                        
                # %%
                try:
                    ber, num_error_bits, success, num_info_error_bits, text_decoded, patt_info = get_per_image_ecc_bit_error_rate(imfile, needle_size, needlenet, mask=mask, rm_order=rm_order,save_fig=save_fig, fig_subdir=fig_subdir, known_patt_info=known_patt_info, patt_str=patt_str, error_after_transform=error_after_transform, no_flip=no_flip, bin_method=bin_method, binnet=binnet)

                    if success:
                        if image_mode:
                            file_row = [fname, patt_info, 'SUCCESS: '+text_decoded, '{}'.format(
                                num_error_bits), '{:.2%}'.format(ber), '{}'.format(num_info_error_bits), imfile]
                            csvwriter.writerow(file_row)
                        else:
                            last_success = True
                            if last_minneb < 0 or last_mininfoneb > num_info_error_bits:
                                last_minfname = fname
                                last_originfo = patt_info
                                last_decinfo = text_decoded
                                last_minneb = num_error_bits
                                last_minber = ber
                                last_mininfoneb = num_info_error_bits
                    elif text_decoded: # failure because of mismatch between decoded text and original text
                        if image_mode:
                            if known_patt_info:
                                file_row = [fname, patt_info, 'FAILED: '+text_decoded, '{}'.format(
                                    num_error_bits), '{:.2%}'.format(ber), '{}'.format(num_info_error_bits), imfile]
                            else:
                                file_row = [fname, patt_info, 'FAILED: '+text_decoded, 'FAILED: {}'.format(
                                    num_error_bits), 'FAILED: {:.2%}'.format(ber), 'FAILED: {}'.format(num_info_error_bits), imfile]
                            csvwriter.writerow(file_row)
                        elif last_minneb < 0 or last_mininfoneb > num_info_error_bits:
                            last_minfname = fname
                            last_originfo = patt_info
                            last_decinfo = text_decoded
                            last_minneb = num_error_bits
                            last_minber = ber
                            last_mininfoneb = num_info_error_bits
                    else: # failure because of Reed-Muller decoding failure
                        if image_mode:
                            if known_patt_info:
                                file_row = [fname, patt_info, 'FAILED (Reed-Muller decoding)', '{}'.format(
                                    num_error_bits), '{:.2%}'.format(ber), '{}'.format(num_info_error_bits), imfile]
                            else:
                                file_row = [fname, patt_info, 'FAILED (Reed-Muller decoding)', 'FAILED: {}'.format(
                                    num_error_bits), 'FAILED: {:.2%}'.format(ber), 'FAILED: {}'.format(num_info_error_bits), imfile]
                            csvwriter.writerow(file_row)
                        elif last_minneb < 0 or last_mininfoneb > num_info_error_bits:
                            last_minfname = fname
                            last_originfo = patt_info
                            last_decinfo = text_decoded
                            last_minneb = num_error_bits
                            last_minber = ber
                            last_mininfoneb = num_info_error_bits
                except:
                    print('    Failed to get error bits from %s.' % imfile)

                    if image_mode:
                        file_row = [fname,'FAILED', 'FAILED (initial rectification)', 'FAILED', 'FAILED', 'FAILED', imfile]
                        csvwriter.writerow(file_row)
        if subdirlist and not image_mode: # not empty list
            # record last-ever instance
            if last_success: # update successful recognization
                file_row = [last_minfname, last_originfo, 'SUCCESS: '+last_decinfo, '{}'.format(last_minneb), '{:.2%}'.format(last_minber), '{}'.format(last_mininfoneb), last_instancepath]
            elif last_minber >= 0: # update last instance w/ any recognizable images
                file_row = [last_minfname, last_originfo, 'FAILED: '+last_decinfo, '{}'.format(last_minneb), '{:.2%}'.format(last_minber), '{}'.format(last_mininfoneb), last_instancepath]
            else: # update last instance w/ no recognizable images
                file_row = [last_minfname, last_originfo, 'FAILED', 'FAILED', 'FAILED', 'FAILED', last_instancepath]
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
    parser.add_argument("--unknown_patt_info", "-u", action='store_true', default=False)
    parser.add_argument("--patt_str", "-p", type=str, default=PATT_STR)
    parser.add_argument("--image_mode", "-i", action='store_true', default=False)
    parser.add_argument("--rm_order", "-r", type=int, default=RM_ORDER)
    parser.add_argument("--error_after_transform", "-t",
                        action='store_true', default=False)
    parser.add_argument("--no_flip", "-l",
                        action='store_true', default=False)
    parser.add_argument("--bin_method", "-b",
                        type=str, default=BIN_METHOD)
    parser.add_argument("--binnet_num_training", "-m", type=int, default=BINNET_NUM_TRAINING)
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
    get_ecc_bit_error_rate(dirfile, opts.needle_size,
                           needlenet, opts.dir+'/'+opts.csv_file,
                           num_training=opts.num_training,
                           file_ext=opts.file_ext, 
                           save_fig=opts.save_fig,
                           fig_subdir=opts.fig_subdir,
                           known_patt_info=not opts.unknown_patt_info,
                           patt_str=opts.patt_str,
                           image_mode=opts.image_mode,
                           rm_order=opts.rm_order, 
                           error_after_transform=opts.error_after_transform, 
                           no_flip=opts.no_flip,
                           bin_method=opts.bin_method,
                           binnet=binnet)
    end = time.time()
    print(f"\n\nTotal runtime (excluding loading NeedleNet) is {end - start} seconds\n")
    
if __name__ == "__main__":
    main()
