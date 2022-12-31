import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import batch_norm_backward_elemt, bilinear, optim
from torch.utils.data import DataLoader, random_split

import wandb
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from binnet import BinNet
from binnet_dataset import BinNetDataset
from evaluate_binnet import evaluate_binnet
from utils import dice_loss

DATASET_PATH   = '../sim/output/needle10x10_100k_bin.mat'
CHECKPOINT_DIR = './checkpoints/binnet_10x10'

USE_CORNER_MASK = False
ADD_IMPULSE_NOISE = False

def train_binnet(binnet, device, 
                 dataset_path : str = DATASET_PATH, 
                 checkpoint_dir: str = CHECKPOINT_DIR,
                 epochs : int = 10,
                 batch_size : int = 8,
                 num_workers : int = 8,
                 learning_rate : float = 0.001,
                 val_percent : float = 0.1,
                 save_checkpoint : bool = True,
                 amp_enable: bool = True, 
                 use_dice_loss: bool = False,
                 use_corner_mask: bool = USE_CORNER_MASK,
                 add_impulse_noise: bool = ADD_IMPULSE_NOISE):
    # [1] create BinNet Dataset
    dataset = BinNetDataset(dataset_path, use_corner_mask, add_impulse_noise)    

    # [2] split whole dataset into train / validation parts [val_percent]
    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val

    train_set, val_set = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(0))

    # [3] train / validation dataloaders
    loader_args = dict(batch_size = batch_size, num_workers = num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    # [4] logging and wandb tracker
    tracker = wandb.init(project='BinNet', resume='allow', entity='yliu12')
    tracker.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent, save_checkpoint=save_checkpoint, amp_enable=amp_enable))

    logging.info(f'''Starting training:
        Epochs:            {epochs}
        Batch size:        {batch_size}
        Learning rate:     {learning_rate}
        Training size:     {num_train}
        Validation size:   {num_val}
        Checkpoints:       {save_checkpoint}
        Device:            {device.type}
        Mixed Precision:   {amp_enable}
        Use Dice Loss:     {use_dice_loss}
        Use corner mask:   {use_corner_mask}
        Add impulse noise: {add_impulse_noise}
        Dataset path:      {dataset_path}
        Checkpoint dir:    {checkpoint_dir}
    ''')

    # [5] setting up training environment - optimizer, loss, learning rate scheduler, loss scaling for AMP
    # optimizer = optim.SGD(binnet.parameters(), lr=0.005, momentum=0.9) # SGD
    # optimizer = optim.SGD(binnet.parameters(), lr=0.005, weight_decay=5e-8, momentum=0.9) # SGD
    optimizer = optim.RMSprop(binnet.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9) #
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.CrossEntropyLoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enable) # AMP

    step_global = 0

    # [6] start training
    for epoch in range(epochs):
        binnet.train()
        loss_epoch = 0
        with tqdm(total=num_train, desc=f'epoch {epoch+1}/{epochs}', unit='image') as pbar: # progress bar from tqdm
            for idx, (image, mask) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                image = image.to(device=device, dtype=torch.float32) # [B,1,H,W]
                mask  = mask.to(device=device, dtype=torch.long)     # [B,H,W]
                
                with torch.cuda.amp.autocast(enabled=amp_enable):
                    mask_pred = binnet(image)

                    # print(image.shape, mask.shape, mask_pred.shape)
                    if use_dice_loss:
                        loss = criterion(mask_pred, mask) \
                            + dice_loss(F.softmax(mask_pred, dim=1).float(), F.one_hot(mask, binnet.out_channels).permute(0,3,1,2).float(),multiclass=True)
                    else:
                        loss = criterion(mask_pred, mask)
                
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
                    for tag, value in binnet.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    # [TODO] evaluation and visualization of the four masks
                    val_score = evaluate_binnet(binnet, val_loader, device)
                    scheduler.step(val_score)
                    
                    logging.info('Validation Dice: {}'.format(val_score))
                    tracker.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_score,
                        'image': wandb.Image(image[0].cpu()),
                        'masks': {
                            'true': wandb.Image(mask[0].float().cpu()),
                            'pred': wandb.Image(1-torch.softmax(mask_pred, dim=1)[0].float().cpu()),
                        },
                        'step': step_global,
                        'epoch': epoch,
                        **histograms
                    })
        
        # save checkpoint for each epoch
        if save_checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(binnet.state_dict(), str(Path(checkpoint_dir) / 'binnet_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'checkpoint BinNet epoch #{epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch BinNet Training (image binarization)')

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
    parser.add_argument('--no-amp', dest='no_amp', action='store_true',      
                        default=False, help='Not using mixed precision')
    parser.add_argument('--use-dice-loss', dest='use_dice_loss', 
                        action='store_true', default=False, help='Use dice loss')
    parser.add_argument('--use-corner-mask', dest='use_corner_mask',
                        action='store_true', default=USE_CORNER_MASK, help='Use corners to generate masks as binarized images')
    parser.add_argument('--add-impulse-noise', dest='add_impulse_noise',
                        action='store_true', default=ADD_IMPULSE_NOISE, help='Add impulse noise for training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    binnet = BinNet(in_channels=1, out_channels=2, bilinear=False)

    # if torch.cuda.device_count() > 1:
    #     logging.info(f'  {torch.cuda.device_count()} GPUs in use')
    #     binnet = nn.DataParallel(binnet)
    binnet.to(device=device)

    logging.info(f'Binarization Network:\n'
                 f'\t{binnet.in_channels} input channel(s)\n'
                 f'\t{binnet.out_channels} output channels/classes\n'
                 f'\t{binnet.bilinear} bilinear upsampling\n')
    
    if args.load:
        binnet.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    try:
        train_binnet(binnet=binnet, device=device, 
                     dataset_path=args.dataset_path,
                     checkpoint_dir=args.checkpoint_dir,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     learning_rate=args.lr,
                     val_percent=args.val / 100.,
                     save_checkpoint=True,
                     amp_enable=not args.no_amp,
                     use_dice_loss=args.use_dice_loss,
                     use_corner_mask=args.use_corner_mask,
                     add_impulse_noise=args.add_impulse_noise)
    except KeyboardInterrupt:
        torch.save(binnet.state_dict(), 'binnet_interrupted.pth')
        logging.info('training BinNet interrupted and intermediate model saved.')
        raise
