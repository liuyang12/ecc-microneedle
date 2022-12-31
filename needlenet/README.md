# Microneedle vision (global solution) and error correction system

## [0] pre-configuration [omit this if no missing dependencies]

### `conda` virtual environment for dependencies
1. `conda create`
2. `conda activate needlenet`

## [1] Apply NeedleNet for recognition and calculate statstics

### 1. For an indivisual 96-bit image
Open Jupyter Notebook `needlenet_bit_error_rates.ipynb` and modify `imfile` (and `needle_size` for 12x12 microneedle arrays)  

### 2. For a group of 96-bit images in a folder
Go to the `scripts` folder by typing `cd ./scripts` 
for example: cd C:\Users\jooli\Dropbox (MIT)\Vaccine_Tracking2\ecc-microneedle\needlenet\scripts

```
python needlenet_bit_error_rates.py --dir FOLDER_PATH
```
for example:
```
python needlenet_bit_error_rates.py --dir "C:\Users\jooli\Dropbox (MIT)\Vaccine_Tracking2\Real Images\Real Images\Jooli's\96bit_new code\Dakar\Big QD"
```
By adding `--save_fig` or `-f`, one can save all the figures (with 1. raw input image; 2. cropped input image; 3. recognized binary microneedle array before correcting orientation; 4. recognized binary microneedle array after correcting orientation; 5. Error bits; 6. original blank microneedle array) in the corresponding sub-folder `fig`.
For example 
```
python needlenet_bit_error_rates.py --dir "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/96-bit MNP/Gin-1pyr-APPL3" -f
```
### 3. For a group of 96-bit images of 1 patch [TODO] d

### 4. For an individual patterned image [both 10x10 and 12x12 microneedle arrays]
Open Jupyter Notebook `needlenet_error_correction.ipynb` and modify `imfile` (and `needle_size` for 12x12 microneedle arrays)  

### 5. For a group of patterned images in a folder [both 10x10 and 12x12 microneedle arrays]
```
python needlenet_error_correction.py --dir FOLDER_PATH
```
for example [10x10 microneedle array by default]:
```
python needlenet_error_correction.py --dir "C:\Users\jooli\Dropbox (MIT)\Vaccine_Tracking2\Real Images\Real Images\Jooli's\Rat study\5) Group K\2) Booster" --no_flip
```
and 12x12 microneedle array
```
python needlenet_error_correction.py --needle_size 12 --dir "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/12x12 Pattern/"
```
and 17x17 microneedle array
```
python needlenet_error_correction.py --needle_size 17 --dir "C:\Users\jooli\Dropbox (MIT)\Vaccine_Tracking2\Real Images\Real Images\Jooli's\Rat study\1) group D_OPMR only 17x17\17 Sept 2" --no_flip
```
#### Useful Options
- `--needle_size` or `-a` Specify the side size of the microneedle patch
- `--dir` or `-d` Specify the folder containing all the images to be analyzed and where the resultant `.csv` file will be stored
- `--no_flip` or `-l` DO NOT flip the image horizontally for 17x17 microneedle patches
- `--bin_method` or `-b` Binarization method - `binnet` for Binarization Net [default] and `thresh` for threshold-based binarization [legacy]
- `--file_ext` or `-e` Specify the file extension of images to be analyzed, default is `jpg`.
- `--save_fig` or `-f` Save the intermediate results of each image in the corresponding subfolder. default is OFF
- `--image_mode` or `-i` Results in `.csv` file is in an image-wise manner instead of instance-wise manner [default].
- `--num_training` or `-n` Number of training samples for recognition network, default is 650(k).
- `--binnet_num_training` or `-m` Number of training samples for binnarization network, default is 400(k).


Similar to Part 2, by adding `--save_fig` or `-f`, one can save all the figures (with 1. raw input image; 2. cropped input image; 3. recognized binary microneedle array before correcting orientation; 4. recognized binary microneedle array after correcting orientation; 5. Error bits; 6. original microneedle array [after masking]) in the corresponding sub-folder `fig`.
For example 
```
python needlenet_error_correction.py --needle_size 12 --dir "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/12x12 Pattern/" -f
```

Training number 150k, saved csv file _150.csv
```
python needlenet_error_correction.py --needle_size 12 --dir "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/12x12 Pattern/" -n 150 -c needlenet_ecc_bit_error_rates_150k.csv
```

## [2] Train NeedleNet - For re-training only [no need for deployment]
### NeedleNet 10x10
```
python train.py --amp --dir_img ../sim/output/needle10x10_image120x120/image/ --dir_mask ../sim/output/needle10x10_image120x120/mask/ --dir_ckpt ./ckpt_10x10_100k --needle-size 10 -e 10 -b 2 -w 8 

python train.py --amp --dir_img ../sim/output/needle10x10_image120x120_all_500k/image/ --dir_mask ../sim/output/needle10x10_image120x120_all_500k/mask/ --dir_ckpt ./ckpt_10x10_500k --needle-size 10 -e 5 -b 2 -w 16

python train.py --amp --dir_img /data/yliu/proj/microneedle/data/needle10x10_image120x120_all_500k/image/ --dir_mask /data/yliu/proj/microneedle/data/needle10x10_image120x120_all_500k/mask/ --dir_ckpt ./ckpt_10x10_500k --needle-size 10 -e 5 -b 2 -w 8

python train.py --amp --dir_img /ssd/home/yliu/data/needle10x10_image120x120_all_500k/image/ --dir_mask /ssd/home/yliu/data/needle10x10_image120x120_all_500k/mask/ --dir_ckpt ./ckpt_10x10_500k --needle-size 10 -e 5 -b 2 -w 8

python train.py --amp --dir_img /data/yliu/proj/microneedle/data/needle10x10_image120x120_dim_50k/image/ --dir_mask /data/yliu/proj/microneedle/data/needle10x10_image120x120_dim_50k/mask/ --load ./checkpoints/model_10x10_100k.pth --dir_ckpt ./checkpoints/ckpt_10x10_150k --needle-size 10 -e 10 -b 2 -w 8 
```

single .mat file dataset
```
# itachi
python train.py --amp --dir_dataset /data/yliu/proj/microneedle/data/needle10x10_image120x120_ord_100k_recog.mat --dir_ckpt ./ckpt_10x10_100k_mat --needle-size 10 -e 10 -b 2 -w 8


# graphics server
python train.py --amp --dir_dataset ../sim/output/needle10x10_image120x120_ord_100k_recog.mat --dir_ckpt ./ckpt_10x10_100k_mat --needle-size 10 -e 10 -b 4 -w 16

python train.py --amp --dir_dataset ../sim/output/needle10x10_image120x120_ord_100k_recog.mat --load ./checkpoints/model_10x10_150k.pth --dir_ckpt ./checkpoints/ckpt_10x10_250k_mat --needle-size 10 -e 5 -b 64 -w 16

python train.py --amp --dir_dataset ../sim/output/needle10x10_300k_recog.mat --load ./checkpoints/model_10x10_350k.pth --dir_ckpt ./checkpoints/ckpt_10x10_650k_crop --needle-size 10 -e 5 -b 64 -w 16

python train.py --amp --dir_dataset /data/yliu/proj/microneedle/data/needle10x10_image120x120_ord_500k_recog.mat --load ./checkpoints/model_10x10_200k.pth --dir_ckpt ./checkpoints/ckpt_10x10_700k_mat --needle-size 10 -e 5 -b 64 -w 16
```

### NeedleNet 12x12
```
python train.py --amp --dir_img ../sim/output/needle12x12_image150x150/image/ --dir_mask ../sim/output/needle12x12_image150x150/mask/ --dir_ckpt ./ckpt_12x12_100k --needle-size 12 -e 10 -b 2 -w 8 

python train.py --amp --dir_img ../sim/output/needle12x12_image144x144_all_500k/image/ --dir_mask ../sim/output/needle12x12_image144x144_all_500k/mask/ --dir_ckpt ./ckpt_12x12_500k --needle-size 12 -e 5 -b 2 -w 16

python train.py --amp --dir_img /ssd/home/yliu/data/needle12x12_image144x144_all_500k/image/ --dir_mask /ssd/home/yliu/data/needle12x12_image144x144_all_500k/mask/ --dir_ckpt ./ckpt_12x12_500k --needle-size 12 -e 5 -b 2 -w 8

python train.py --amp --dir_img /data/yliu/proj/microneedle/data/needle12x12_image144x144_dim_50k/image/ --dir_mask /data/yliu/proj/microneedle/data/needle12x12_image144x144_dim_50k/mask/ --load ./checkpoints/model_12x12_100k.pth --dir_ckpt ./checkpoints/ckpt_12x12_150k --needle-size 12 -e 10 -b 2 -w 8
```


single .mat file
```
python train.py --amp --dir_dataset /ssd/home/yliu/data/needle12x12_500k_recog.mat --load ./checkpoints/model_12x12_150k.pth --dir_ckpt ./checkpoints/ckpt_12x12_650k --needle-size 12 -e 5 -b 64 -w 16
```


### NeedleNet 17x17
```
python train.py --amp --dir_img /data/yliu/proj/microneedle/data/needle17x17_image176x176_all_100k/image/ --dir_mask /data/yliu/proj/microneedle/data/needle17x17_image176x176_all_100k/mask/ --dir_ckpt ./ckpt_17x17_100k --needle-size 17 -e 10 -b 2 -w 8 

python train.py --amp --dir_img /data/yliu/proj/microneedle/data/needle17x17_image176x176_dim_50k/image/ --dir_mask /data/yliu/proj/microneedle/data/needle17x17_image176x176_dim_50k/mask/ --load ./checkpoints/model_17x17_100k.pth --dir_ckpt ./checkpoints/ckpt_17x17_150k --needle-size 17 -e 10 -b 2 -w 8 

```

## Prediction 

### NeedleNet 10x10
```
python predict.py -m model_10x10.pth -f --needle-size 10 -i ./data/test_sim

python predict.py -m model_10x10.pth -f --needle-size 10 -i ../../microneedle/data/Images0706/cropped_120x120

python predict.py -m model_10x10.pth -f --needle-size 10 -i "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/10x10 Pattern/cropped_120x120"

python predict.py -m model_10x10_100k.pth -f --needle-size 10 -i "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/10x10 Pattern/cropped_120x120"
```

### NeedleNet 12x12
```
python predict.py -m model_12x12.pth -f --needle-size 12 -i ./data/test_sim

python predict.py -m model_12x12.pth -f --needle-size 12 -i "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/12x12 Pattern/cropped_136x136"

python predict.py -m model_12x12_100k.pth -f --needle-size 12 -i "/data/yliu/docs/Dropbox (MIT)/Vaccine_Tracking2/Real Images/Real Images/Jooli's/12x12 Pattern/cropped_136x136"
```


## Data generation

in SuperCloud

```
LLsub matlab_generate10x10_100k.sh
```


## Train RectNet
```
python train_rectnet.py -p /data/yliu/proj/microneedle/data/needle10x10_image120x120_ord_100k_rect.mat -c ./checkpoints/rectnet_10x10 -e 10 -b 64 -w 16

python train_rectnet.py -p /data/yliu/proj/microneedle/data/needle10x10_image120x120_dim_250k_rect.mat -f ./checkpoints/rectnet_10x10_500k.pth -c ./checkpoints/rectnet_10x10_750k -e 5 -b 64 -w 16

python train_rectnet.py -p ../sim/output/needle10x10_image120x120_ord_500k_rect.mat -c ./checkpoints/rectnet_10x10 -e 10 -b 64 -w 16

python train_rectnet.py -p /data/yliu/proj/microneedle/data/needle10x10_image120x120_ord_500k_rect.mat -c ./checkpoints/rectnet_10x10_500k_xy -e 10 -b 64 -w 16 --xy
```

## Train BinNet
```
python train_binnet.py -p /data/yliu/proj/microneedle/data/needle10x10_100k_bin.mat -c ./checkpoints/binnet_10x10 -e 10 -b 16 -w 16

python train_binnet.py -p /data/yliu/proj/microneedle/data/needle10x10_300k_bin.mat -c ./checkpoints/binnet_10x10_400k -f ./checkpoints/binnet_100k.pth -e 5 -b 16 -w 16

python train_binnet.py -p ../sim/output/needle10x10_100k_bin.mat -c ./checkpoints/binnet_10x10 -e 10 -b 64 -w 16
```