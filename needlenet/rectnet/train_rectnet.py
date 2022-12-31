import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import batch_norm_backward_elemt, optim
from torch.utils.data import DataLoader, random_split

import wandb
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from rectnet import RectNet
from rectnet_dataset import RectNetDataset
from evaluate_rectnet import evaluate_rectnet
from utils import draw_polygon_on_image, crop_homography

DATASET_PATH   = '../sim/output/needle10x10_image120x120_ord_100k_rect.mat'
CHECKPOINT_DIR = './checkpoints/rectnet_10x10'


def train_rectnet(rectnet, device, 
                  dataset_path : str = DATASET_PATH, 
                  checkpoint_dir: str = CHECKPOINT_DIR,
                  epochs : int = 10,
                  batch_size : int = 8,
                  num_workers : int = 8,
                  learning_rate : float = 0.001,
                  val_percent : float = 0.1,
                  save_checkpoint : bool = True,
                  amp_enable : bool = True, 
                  xy_channels : bool = False):
    # [1] create RectNet Dataset
    dataset = RectNetDataset(dataset_path, reorder=True, xy_channels=xy_channels)    

    # [2] split whole dataset into train / validation parts [val_percent]
    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val

    train_set, val_set = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(0))

    # [3] train / validation dataloaders
    loader_args = dict(batch_size = batch_size, num_workers = num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    # [4] logging and wandb tracker
    tracker = wandb.init(project='RectNet', resume='allow', entity='yliu12')
    tracker.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent, save_checkpoint=save_checkpoint, amp_enable=amp_enable, xy_channels=xy_channels))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {num_train}
        Validation size: {num_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp_enable}
        XY Channels:     {xy_channels}
        Dataset path:    {dataset_path}
        Checkpoint dir:  {checkpoint_dir}
    ''')

    # [5] setting up training environment - optimizer, loss, learning rate scheduler, loss scaling for AMP
    # optimizer = optim.SGD(rectnet.parameters(), lr=0.005, momentum=0.9) # SGD
    # optimizer = optim.SGD(rectnet.parameters(), lr=0.005, weight_decay=5e-8, momentum=0.9) # SGD
    optimizer = optim.RMSprop(rectnet.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9) #
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.MSELoss() # MSE
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enable) # AMP

    step_global = 0

    # [6] start training
    for epoch in range(epochs):
        rectnet.train()
        loss_epoch = 0
        with tqdm(total=num_train, desc=f'epoch {epoch+1}/{epochs}', unit='image') as pbar: # progress bar from tqdm
            for idx, (image, corner) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                image  =  image.to(device=device, dtype=torch.float32) # [B,1,H,W]
                corner = corner.to(device=device, dtype=torch.float32) # [B,4*2]

                with torch.cuda.amp.autocast(enabled=amp_enable):
                    corner_pred = rectnet(image)

                    loss = criterion(corner_pred, corner)
                
                grad_scaler.scale(loss).backward() # AMP
                grad_scaler.step(optimizer)        # AMP
                grad_scaler.update()               # AMP

                # progress bar update
                pbar.update(image.shape[0])
                step_global  +=  1
                loss_epoch   +=  loss.item()
                tracker.log({
                    'training loss' : loss.item(),
                    'step'          : step_global, 
                    'epoch'         : epoch  
                })
                pbar.set_postfix(**{'loss (batch)' : loss.item()})

                # evaluation every 1/10 round
                if step_global % (num_train // (10 * batch_size)) == 0:
                    histograms = {}
                    for tag, value in rectnet.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    # [TODO] evaluation and visualization of the four corners
                    val_loss = evaluate_rectnet(rectnet, val_loader, device)
                    scheduler.step(val_loss)
                    
                    img = image[0, 0].cpu().numpy()*255.
                    cor = corner[0].cpu().numpy().reshape((-1,2))
                    cor_p = corner_pred[0].detach().cpu().numpy().reshape((-1, 2))

                    logging.info('Validation loss: {}'.format(val_loss))
                    tracker.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation loss': val_loss,
                        'image': wandb.Image(img),
                        'corners': {
                            'true': wandb.Image(draw_polygon_on_image(img, cor)),
                            'pred': wandb.Image(draw_polygon_on_image(img, cor_p)),
                        },
                        'crop': {
                            'true': wandb.Image(crop_homography(img, cor)),
                            'pred': wandb.Image(crop_homography(img, cor_p)),
                        },
                        'step': step_global,
                        'epoch': epoch,
                        **histograms
                    })
        
        # save checkpoint for each epoch
        if save_checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(rectnet.state_dict(), str(Path(checkpoint_dir) / 'rectnet_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'checkpoint RectNet epoch #{epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch RectNet Training (downsampled raw image -> four corners [top-left; counter-clockwise])')

    parser.add_argument('-p', '--dataset-path', dest='dataset_path', 
                        type=str, default=DATASET_PATH, 
                        help='path of the dataset for training')
    parser.add_argument('-c', '--checkpoint-dir', dest='checkpoint_dir',
                        type=str, default=CHECKPOINT_DIR,
                        help='directory of the saved checkpoints')
    parser.add_argument('-e', '--epochs',metavar='E', 
                        type=int, default=10, 
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', 
                        metavar='B', type=int, default=2, 
                        help='Batch size')
    parser.add_argument('-w', '--num-workers', dest='num_workers', 
                        metavar='W', type=int, default=8,
                        help='Number of workers for dataloading')
    parser.add_argument('-l', '--learning-rate', dest='lr', 
                        metavar='LR', type=float, default=0.00001,
                        help='Learning rate')
    parser.add_argument('-f', '--load', type=str, default=False, 
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val',
                        type=float, default=10.0,
                        help='Percentage of data used as validation (0-100)%')
    parser.add_argument('--no_amp', action='store_true', default=False, 
                        help='Not using mixed precision')
    parser.add_argument('--xy', action='store_true', default=False, 
                        help='Add XY coordinates to the input channels')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    if args.xy:
        in_channels = 3
    else:
        in_channels = 1
    # rectnet = RectNet(in_channels=in_channels,  net="R", avgpool_size=8, fc_size=1024)
    rectnet = RectNet(in_channels=in_channels, net="RH", avgpool_size=16, fc_size=1024)

    logging.info(f'Network {rectnet.net}:\n'
                 f'\t{rectnet.in_channels} input channel(s)\n'
                 f'\t{rectnet.num_points} output 2D points\n'
                 f'\t{rectnet.last_conv_channel} channels in last convolutional layer\n'
                 f'\t{rectnet.avgpool_size} avarage pooling size\n'
                 f'\t{rectnet.fc_size} fully connnected layer size\n')
    
    if args.load:
        rectnet.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    rectnet.to(device=device)

    try:
        train_rectnet(rectnet=rectnet, device=device, 
                      dataset_path=args.dataset_path,
                      checkpoint_dir=args.checkpoint_dir,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      learning_rate=args.lr,
                      val_percent=args.val / 100.,
                      save_checkpoint=True,
                      amp_enable=not args.no_amp,
                      xy_channels=args.xy)
    except KeyboardInterrupt:
        torch.save(rectnet.state_dict(), 'rectnet_interrupted.pth')
        logging.info('training interrupted and intermediate model saved.')
        sys.exit(0)
