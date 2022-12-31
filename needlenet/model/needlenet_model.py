""" Full NeedleNet model via assembling the parts """
""" following `unet_model.py` in https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """

from .needlenet_parts import *

class NeedleNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=1):
        super(NeedleNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes  = num_classes
        self.inc   = Conv(num_channels, 32, kernel_size=3, padding=1)
        self.down1 = Down(32, 64, kernel_size=3, padding=1)
        self.down2 = Down(64, 128, kernel_size=3, padding=0)
        self.down3 = Down(128, 128, kernel_size=5, padding=0)
        self.outc  = OutConv(128, num_classes)

    def forward(self, x):
        x1  = self.inc(x)
        x2  = self.down1(x1)
        x3  = self.down2(x2)
        x4  = self.down3(x3)
        out = self.outc(x4)
        return out
