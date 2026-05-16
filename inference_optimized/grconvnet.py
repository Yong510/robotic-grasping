import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from inference.models.grasp_model import ResidualBlock

class GenerativeResnet(nn.Module):
    """
    轻量级多尺度骨干网络，输出 c2/c3/c4
    """
    def __init__(self, input_channels=4, channel_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.res_blocks = nn.Sequential(*[ResidualBlock(128, 128) for _ in range(5)])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        c2 = F.relu(self.bn2(self.conv2(x)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        c4 = F.relu(self.bn4(self.conv4(c3)))
        c4 = self.res_blocks(c4)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4)])

    def forward(self, x_in):
        return self.extract_features(x_in)
