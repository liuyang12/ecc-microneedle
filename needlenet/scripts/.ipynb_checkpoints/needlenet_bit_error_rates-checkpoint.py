
# %%
# import external packages
import csv
import time
import glob
import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
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
IMAGE_DIR = '/data/yliu/docs/Dropbox(MIT)/Vaccine_Tracking2/Real Images/Real Images/96-bit MNP/'
FILE_EXT = 'jpg'
CSV_FILE = 'needlenet_bit_error_rates.csv'
SAVE_FIG = False
plt.ioff()


def get_per_image_bit_error_rate(imfile, needle_size, needlenet, save_fig=SAVE_FIG):

    # predict binary microneedle array from the raw input image
    im = cv.imread(imfile, 0)
    needle, im_crop = microneedle_array_from_raw_image(im, needle_size, net=needlenet)

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
    needle = np.flip(needle,1).astype(int)

    # orientation detection (based on the black corner)
    c1 = np.sum(needle[0:p,0:p])   # top-left
    c2 = np.sum(needle[0:p,-p:])   # top-right
    c3 = np.sum(needle[-p:,-p:])
    c4 = np.sum(needle[-p:,0:p])   # bottom-left

    rotnum = np.argmin([c1,c2,c3,c4]) + 3

    binneedle = np.rot90(needle,rotnum)

    errorneedle = binneedle-blank

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
        plt.imshow(np.absolute(errorneedle), cmap='gray')
        plt.title('Error bit rate {:.2%} (white - error; black - correct)'.format(ber))
        plt.subplot(326)
        plt.imshow(blank, cmap='gray')
        plt.title('Full blank microneedle array (original)')
        plt.savefig(add_suffix_filename(imfile, 'fig'), bbox_inches='tight')
        plt.close(fig)

    return ber, num_error_bits


def get_bit_error_rate(imdir, needle_size, needlenet, csvfilename, save_fig=SAVE_FIG):
    # write to csv
    fields = ['File name','Number of error bits','Bit error rate','Full file path']
        
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
                ber, num_error_bits = get_per_image_bit_error_rate(imfile, needle_size, needlenet, save_fig=save_fig)

                file_row = [fname, '{}'.format(num_error_bits), '{:.2%}'.format(ber), imfile]
            except:
                print('    Failed to get error bits from %s.' % imfile)

                file_row = [fname, 'FAILED', 'FAILED', imfile]

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

    # runtime excluding loading the network
    start = time.time()
    get_bit_error_rate(dirfile, opts.needle_size,
                       needlenet, opts.dir+'/'+opts.csv_file,save_fig=opts.save_fig)
    end = time.time()
    print(f"\n\nTotal runtime (excluding loading NeedleNet) is {end - start} seconds\n")
    
if __name__ == "__main__":
    main()
