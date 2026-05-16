import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from grconvnet import GenerativeResnet
from fpn import FPN

class MFENet(nn.Module):
    def __init__(self, input_channels=4, dropout=0.1, fpn_channels=128):
        super().__init__()
        self.encoder = GenerativeResnet(input_channels=input_channels)
        self.fpn = FPN(OrderedDict([("c2", 64), ("c3", 128), ("c4", 128)]), out_channels=fpn_channels)
        self.up = nn.ConvTranspose2d(fpn_channels, 64, kernel_size=4, stride=2, padding=1)
        self.up_bn = nn.BatchNorm2d(64)
        self.refine = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.refine_bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=dropout)
        self.pos_output = nn.Conv2d(32,1,kernel_size=1)
        self.cos_output = nn.Conv2d(32,1,kernel_size=1)
        self.sin_output = nn.Conv2d(32,1,kernel_size=1)
        self.width_output = nn.Conv2d(32,1,kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        features = self.encoder.extract_features(x)
        p_features = self.fpn(features)
        x = p_features["p2"]
        x = F.relu(self.up_bn(self.up(x)))
        x = F.relu(self.refine_bn(self.refine(x)))
        x = self.dropout(x)
        return (self.pos_output(x), self.cos_output(x), self.sin_output(x), self.width_output(x))
