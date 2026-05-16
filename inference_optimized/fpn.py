import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FPN(nn.Module):
    def __init__(self, in_channels_dict, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleDict({
            name: nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for name, in_ch in in_channels_dict.items()
        })
        self.smooth_convs = nn.ModuleDict({
            name: nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for name in in_channels_dict.keys()
        })
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        c2, c3, c4 = features["c2"], features["c3"], features["c4"]
        p4 = self.lateral_convs["c4"](c4)
        p3 = self.lateral_convs["c3"](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs["c2"](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.smooth_convs["c4"](p4)
        p3 = self.smooth_convs["c3"](p3)
        p2 = self.smooth_convs["c2"](p2)
        return OrderedDict([("p2", p2), ("p3", p3), ("p4", p4)])
