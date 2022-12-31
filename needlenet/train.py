import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, NeedleDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
# from unet import UNet
from model import NeedleNet

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_img = Path('../sim/output/needle10x10_image120x120/image/')
# dir_mask = Path('../sim/output/needle10x10_image120x120/mask/')

dir_dataset = Path('../sim/output/needle10x10_image120x120/')
dir_checkpoint = Path('./ckpt_10x10/')


def train_net(net,
              device,
              dir_dataset,
              dir_ckpts,
              epochs: int = 5,
              batch_size: int = 2,
              num_workers: int = 8,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              needle_size: int = 10,
              amp: bool = False,
              use_dice_loss: bool = False):
    # 1. Create dataset
    try:
        dataset = NeedleDataset(dir_dataset, img_scale, needle_size)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_dataset, img_scale, needle_size)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='NeedleNet', resume='allow', anonymous='must')
    experiment = wandb.init(project='NeedleNet', resume='allow', entity='yliu12')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                # print(images.shape, true_masks.shape)
                # print(torch.min(true_masks), torch.max(true_masks))

                assert images.shape[1] == net.num_channels, \
                    f'Network has been defined with {net.num_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    # print(torch.min(true_masks), torch.max(true_masks))
                    # print(torch.min(masks_pred), torch.max(masks_pred))
                    # print(images.shape, true_masks.shape, masks_pred.shape)
                    if use_dice_loss:
                        loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.num_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
                    else:
                        loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % (n_train // (10 * batch_size)) == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(net, val_loader, device)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_score,
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(1-torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

        if save_checkpoint:
            Path(dir_ckpts).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(
                Path(dir_ckpts) / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch NeedleNet Training (rectified image -> binary needle patch)')
    parser.add_argument('--dir_dataset', '-i', type=str,
                        default=dir_dataset, help='directory of the dataset (a folder [legacy] or a single file [efficient])')
    parser.add_argument('--dir_ckpt', '-c', type=str,
                        default=dir_checkpoint, help='directory of the saved checkpoints')
    parser.add_argument('--needle-size', '-d', dest='needle_size',
                        metavar='N', type=int, default=10, help='Size of the microneedle array (10x10 or 12x12 or 17x17)')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--num-workers', '-w', dest='num_workers',metavar='W', type=int, default=8, help='Number of workers for dataloading')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--dice-loss', dest='dice_loss', action='store_true', default=False, help='Use dice loss')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = UNet(n_channels=3, n_classes=2, bilinear=True)

    net = NeedleNet(num_channels=1, num_classes=2)

    logging.info(f'Network:\n'
                 f'\t{net.num_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  dir_dataset=args.dir_dataset,
                  dir_ckpts=args.dir_ckpt,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  needle_size=args.needle_size,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  use_dice_loss=args.dice_loss)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
