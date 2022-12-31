import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Dict, Any, Optional, cast

'''
Rectification network for downsampled (120x120 px) raw input images
by outputing 

code refernce: https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html
'''
class RectNet(nn.Module):
    def __init__(self, in_channels:int=1, num_points: int = 4, init_weights: bool = True, dropout: float = 0.5, net: str = "R", avgpool_size: int = 8, fc_size: int = 1024) -> None:
        '''
        in_channels  - input channels [int] [default:1]
        num_points   - number of output points [int] [default:4]
        init_weights - initialize network module weights [bool] [default:True]
        dropout      - dropout weight of fully connected layers [float] [default:0.5]
        '''
        super(RectNet, self).__init__()

        # [1] features part - conv nets/batch norm/relu/max pooling
        # RectNet feature conv layers
        net_cfg = cfgs[net]
        self.features = make_layers(net_cfg, in_channels=in_channels, batch_norm=True)

        # [2] adaptive average pooling -> target 2D sizes to rectifier
        self.avgpool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))

        # [3] rectifier -> 4 corner points for rectification (homography)
        last_conv_channel = net_cfg[-2] # number of channels of last conv layer
        self.rectifier = nn.Sequential(
            nn.Linear(last_conv_channel * avgpool_size * avgpool_size, fc_size),
            nn.ReLU(True), 
            nn.Dropout(p=dropout), 
            nn.Linear(fc_size, num_points*2)
        )

        # [0] weights initialization
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        self.in_channels = in_channels
        self.num_points = num_points
        self.init_weights = init_weights
        self.dropout = dropout
        self.net = net
        self.last_conv_channel = last_conv_channel
        self.avgpool_size = avgpool_size
        self.fc_size = fc_size

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.rectifier(x)

        return x


def make_layers(cfg: List[Union[str, int]], in_channels: int = 1, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=False)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    # Rectification net configuration [1x120x120]
    "R":  [32, "M", 64, "M", 64, "M", 128, "M"],
    "R2": [32, "M", 64, "M", 64, 64, "M", 128, 128, "M"],
    "RH": [64, 64, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M"],
    # VGG-Net configurations [3x224x224] https://arxiv.org/pdf/1409.1556.pdf
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

