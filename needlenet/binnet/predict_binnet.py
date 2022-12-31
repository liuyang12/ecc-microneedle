import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from binnet import BinNet

CHECKPOINT = '../checkpoints/binnet_400k.pth'
IN_CHANNELS  = 1 # [fixed] number of input channels for BinNet
OUT_CHANNELS = 2 # [fixed] number of output channels / classes for BinNet
BILINEAR     = False # [fixed ]
BIN_SIZE     = [256, 256] # input size for binarization


def load_binnet(ckpt:str=CHECKPOINT, in_channels:int=IN_CHANNELS, out_channels:int=OUT_CHANNELS, bilinear:bool=BILINEAR):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    binnet = BinNet(in_channels=in_channels, out_channels=out_channels, bilinear=bilinear)

    binnet.to(device=device)

    binnet.load_state_dict(torch.load(ckpt, map_location=device))

    binnet.eval()

    return binnet

def predict_binarization_from_raw_image(im, binnet, device=None, norm_max=True, MAXB=256.):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if norm_max:
        im = im / np.max(im)
    else:
        im = im / (MAXB-1)

    im = torch.from_numpy(im).unsqueeze(0).unsqueeze(0)
    im = im.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask = binnet(im)

        probs = F.softmax(mask, dim=1)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

        # print(full_mask.shape)
    return np.argmax(F.one_hot(full_mask.argmax(dim=0), binnet.out_channels).permute(2, 0, 1).numpy(), axis=0)
