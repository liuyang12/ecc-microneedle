
# %%
# import external packages
import csv
import time
import glob
import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
from reedmuller import reedmuller  # Reed-Muller code
import matplotlib.pyplot as plt

# import internal packages
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from predict import (load_needlenet, microneedle_array_from_raw_image)
from utils.utils import add_suffix_filename

NEEDLE_SIZE = 10
NUM_TRAINING = 100 # k
IMAGE_DIR = '/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli\'s/10x10 Pattern/Applicator 3/'
FILE_EXT = 'jpg'
CSV_FILE = 'needlenet_ecc_bit_error_rates.csv'
SAVE_FIG = False
plt.ioff()

def get_per_image_ecc_bit_error_rate(imfile, needle_size, needlenet, save_fig=SAVE_FIG):

    # predict binary microneedle array from the raw input image
    im = cv.imread(imfile, 0)

    needle, im_crop = microneedle_array_from_raw_image(im, needle_size, net=needlenet)

    # [0] layout of the microneedle array
    #     only use 64 bits for encoding RM(1,8) or [64,7,32]_2-code
    if needle_size == 10:
        corner_size = 3
        corner_hole = True
        rm_order = 1
        use_mask = False
    elif needle_size == 12:
        corner_size = 2
        corner_hole = False
        rm_order = 2
        use_mask = True
    else:
        corner_size = 2
        corner_hole = False
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

    # [2] Reed-Muller encoder
    totalbits = s*s - 4*p*p   # total number of bits for encoding

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
    idxmat[0:p, 0:p] = -1   # top-left
    idxmat[-p:, 0:p] = -1   # bottom-left
    idxmat[0:p, -p:] = -1   # top-right
    idxmat[-p:, -p:] = -1   # bottom-right
    idxvec = idxmat[idxmat > 0]

    # load mask if using a mask for encoding
    if use_mask:
        import scipy.io as sio
        maskfile = f'../checkpoints/mask_{needle_size}x{needle_size}.mat'
        if os.path.isfile(maskfile):
            data = sio.loadmat(maskfile)
            mask = data['mask']
            print(f'  loaded mask file {maskfile}')
        else:
            # add a random bianry mask
            mask = np.random.randint(2, size=(s, s))
            mask[0:p, 0:p] = 0   # top-left
            mask[-p:, 0:p] = 0   # bottom-left
            mask[0:p, -p:] = 0   # top-right
            mask[-p:, -p:] = 0   # bottom-right
            sio.savemat(maskfile, {'mask': mask})
            print(f'  regenerated and saved mask file {maskfile}')

    # flip the binary
    noisyneedle = np.flip(needle, 1).astype(int)

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


    # [3.2] assign value to each bit in the microneedle
    origneedle = blank.copy()

    # blank array with all zeros (binary 0/1 )
    infomat = np.ones((s, s), dtype=bool)
    infomat[0:p, 0:p] = 0   # top-left
    infomat[-p:, 0:p] = 0   # bottom-left
    infomat[0:p, -p:] = 0   # top-right
    infomat[-p:, -p:] = 0   # bottom-right

    # needle[idxvec] = code
    origcode = rmcode.encode(y)
    for i in range(n):
        origneedle[idxvec[i]//s, idxvec[i] % s] = origcode[i]

    if use_mask:
        origneedle = np.logical_xor(origneedle, mask).astype(int)

    errorneedle = noisyneedle_mask-origneedle.astype(int)

    num_error_bits = np.count_nonzero(errorneedle)  # number of flipped pixel bits
    num_info_error_bits = np.count_nonzero(
        errorneedle*infomat)  # number of flipped pixel bits

    ber = num_error_bits/(s*s)  # bit error rate
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
        plt.imshow(np.absolute(errorneedle), cmap='gray')
        plt.title('Error bit rate {:.2%} (white - error; black - correct)'.format(ber))
        plt.savefig(add_suffix_filename(imfile, 'fig'), bbox_inches='tight')
        plt.close(fig)

    return ber, num_error_bits, success, num_info_error_bits, text_dec


def get_ecc_bit_error_rate(imdir, needle_size, needlenet, csvfilename, save_fig=SAVE_FIG):
    # write to csv
    fields = ['File name','Decoded info','Number of error bits','Bit error rate','Number of info error bits (excluding four corners)','Full file path']
        
    # writing to csv file 
    with open(csvfilename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fields)

        imfilelist = glob.glob(imdir,recursive=True)
        print(len(imfilelist))

        for imfile in imfilelist:
            fstem = Path(imfile).stem
            fname = Path(imfile).name

            print(imfile)

            # %%
            try:
                ber, num_error_bits, success, num_info_error_bits, text_decoded = get_per_image_ecc_bit_error_rate(imfile, needle_size, needlenet, save_fig=save_fig)

                if success:
                    file_row = [fname, text_decoded, '{}'.format(num_error_bits), '{:.2%}'.format(ber), '{}'.format(num_info_error_bits), imfile]
                else:
                    file_row = [fname, 'FAILED: '+text_decoded, 'FAILED: {}'.format(
                        num_error_bits), 'FAILED: {:.2%}'.format(ber), 'FAILED: {}'.format(num_info_error_bits), imfile]
            except:
                print('    Failed to get error bits from %s.' % imfile)

                file_row = [fname,'FAILED', 'FAILED', 'FAILED', 'FAILED', imfile]

            csvwriter.writerow(file_row)   
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--needle_size", "-a", type=int, default=NEEDLE_SIZE)
    parser.add_argument("--num_training", "-n", type=int, default=NUM_TRAINING)
    parser.add_argument("--dir", "-d", type=str, default=IMAGE_DIR)
    parser.add_argument("--file_ext", "-e", type=str, default=FILE_EXT)
    parser.add_argument("--csv_file", "-c", type=str, default=CSV_FILE)
    parser.add_argument("--save_fig", "-f", action='store_true', default=False)
    opts = parser.parse_args()
    
    # dirfile = os.path.abspath(opts.dir)+'/*.'+opts.file_ext
    dirfile = opts.dir+'/**/*.'+opts.file_ext

    num_training = 100  # number of training samples *1k
    model = f'../checkpoints/model_{opts.needle_size}x{opts.needle_size}_{opts.num_training}k.pth'

    # load pre-trained NeedleNet [only once]
    needlenet = load_needlenet(model)
    print(f'loaded NeedleNet {model}')

    # runtime excluding loading the network
    start = time.time()
    get_ecc_bit_error_rate(dirfile, opts.needle_size,
                           needlenet, opts.dir+'/'+opts.csv_file,
                           save_fig=opts.save_fig)
    end = time.time()
    print(f"\n\nTotal runtime (excluding loading NeedleNet) is {end - start} seconds\n")
    
if __name__ == "__main__":
    main()
