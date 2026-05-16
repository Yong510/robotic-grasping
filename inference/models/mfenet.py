import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .grasp_model import GraspModel, ResidualBlock


class MFEBackbone(nn.Module):
    def __init__(self, input_channels=4, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.conv1 = nn.Conv2d(input_channels, c1, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        # Make c4 a lower-resolution semantic layer: H/8, W/8.
        self.conv4 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(c3)

        self.res1 = ResidualBlock(c3, c3)
        self.res2 = ResidualBlock(c3, c3)
        self.res3 = ResidualBlock(c3, c3)
        self.res4 = ResidualBlock(c3, c3)
        self.res5 = ResidualBlock(c3, c3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(x)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        c4 = F.relu(self.bn4(self.conv4(c3)))

        c4 = self.res1(c4)
        c4 = self.res2(c4)
        c4 = self.res3(c4)
        c4 = self.res4(c4)
        c4 = self.res5(c4)

        return OrderedDict([
            ('c2', c2),
            ('c3', c3),
            ('c4', c4),
        ])


class FPNLite(nn.Module):
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        c2, c3, c4 = features['c2'], features['c3'], features['c4']

        p4 = self.lateral_convs['c4'](c4)
        p3 = self.lateral_convs['c3'](c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode='bilinear', align_corners=False
        )
        p2 = self.lateral_convs['c2'](c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode='bilinear', align_corners=False
        )

        p4 = self.smooth_convs['c4'](p4)
        p3 = self.smooth_convs['c3'](p3)
        p2 = self.smooth_convs['c2'](p2)

        return OrderedDict([
            ('p2', p2),
            ('p3', p3),
            ('p4', p4),
        ])


class MFENet(GraspModel):
    """
    MFENet compatible with the original skumra/robotic-grasping training pipeline.

    Default forward output is kept unchanged:
        pos, cos, sin, width

    When return_features=True, the model additionally returns multi-scale features for
    object-centric semantic/risk branches without breaking the original compute_loss().
    """
    def __init__(self, input_channels=4, dropout=False, prob=0.1, channel_size=32):
        super(MFENet, self).__init__()

        base = channel_size
        fpn_channels = base * 4

        self.backbone = MFEBackbone(input_channels=input_channels, base_channels=base)
        self.fpn = FPNLite(
            in_channels_dict=OrderedDict([
                ('c2', base * 2),
                ('c3', base * 4),
                ('c4', base * 4),
            ]),
            out_channels=fpn_channels
        )

        self.up = nn.ConvTranspose2d(fpn_channels, base * 2, kernel_size=4, stride=2, padding=1)
        self.up_bn = nn.BatchNorm2d(base * 2)

        self.refine = nn.Conv2d(base * 2, base, kernel_size=3, stride=1, padding=1)
        self.refine_bn = nn.BatchNorm2d(base)

        self.dropout1 = nn.Dropout2d(p=prob) if dropout else nn.Identity()

        self.pos_output = nn.Conv2d(base, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(base, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(base, 1, kernel_size=1)
        self.width_output = nn.Conv2d(base, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        features = self.backbone.extract_features(x)
        p_features = self.fpn(features)

        p2 = p_features['p2']
        x = F.relu(self.up_bn(self.up(p2)))
        x = F.relu(self.refine_bn(self.refine(x)))
        grasp_feat = self.dropout1(x)

        pos_output = self.pos_output(grasp_feat)
        cos_output = self.cos_output(grasp_feat)
        sin_output = self.sin_output(grasp_feat)
        width_output = torch.sigmoid(self.width_output(grasp_feat))

        if return_features:
            aux_features = {
                'features': features,
                'p_features': p_features,
                'grasp_feat': grasp_feat,
            }
            return pos_output, cos_output, sin_output, width_output, aux_features

        return pos_output, cos_output, sin_output, width_output
