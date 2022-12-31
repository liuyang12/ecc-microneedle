""" Parts of the NeedleNet model """
""" following `unet_parts.py` in https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py """

import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """ Convolutional layer (Convolution -> BatchNorm -> ReLU) * N(=1) """

    def __init__(self, in_channels, out_channels, 
                 num_layers=1, kernel_size=3, padding=1):
        super().__init__()

        layers = []
        for ilayer in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=kernel_size, 
                                    padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.multi_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.multi_conv(x)


class Down(nn.Module):
    """ Downscaling with maxpool then multi-conv """

    def __init__(self, in_channels, out_channels,
                 num_layers=1, kernel_size=3, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_channels, out_channels, num_layers, kernel_size, padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    """ Output Convolution layer """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
