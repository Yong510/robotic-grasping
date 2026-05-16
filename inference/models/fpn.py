import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FPN(nn.Module):
    def __init__(self, in_channels_dict, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        # 为每个输入层级创建一个 1x1 卷积，将通道数统一到 out_channels
        self.lateral_convs = nn.ModuleDict()
        for name, in_ch in in_channels_dict.items():
            self.lateral_convs[name] = nn.Conv2d(in_ch, out_channels, kernel_size=1)
        
        # 上采样层（用于融合更高层特征）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 可选的平滑卷积（对融合后特征进一步处理）
        self.smooth_convs = nn.ModuleDict()
        for name in in_channels_dict.keys():
            self.smooth_convs[name] = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        # features: OrderedDict {'c2': Tensor, 'c3': Tensor, 'c4': Tensor} (从低到高分辨率)
        # 我们假设输入的 keys 是按分辨率从高到低排列的（c2 最高，c4 最低）
        names = list(features.keys())
        
        # 1. 对每个输入做 1x1 卷积，统一通道数
        lateral = OrderedDict()
        for name in names:
            lateral[name] = self.lateral_convs[name](features[name])
        
        # 2. 自顶向下路径 + 横向连接
        # 从最低分辨率开始（c4）
        prev = lateral[names[-1]]  # c4
        results = OrderedDict()
        results[names[-1]] = self.smooth_convs[names[-1]](prev)
        
        # 逆序处理更高分辨率层级（c3, c2）
        for name in reversed(names[:-1]):
            # 上采样上一级特征，并与当前 lateral 相加
            prev_upsampled = self.upsample(prev)
            # 处理尺寸不匹配（由于 stride 可能造成的 1 像素差异）
            if prev_upsampled.shape[2:] != lateral[name].shape[2:]:
                prev_upsampled = F.interpolate(prev_upsampled, size=lateral[name].shape[2:], mode='bilinear', align_corners=True)
            prev = lateral[name] + prev_upsampled
            results[name] = self.smooth_convs[name](prev)
        
        # 注意：results 中的键名与输入一致（c2, c3, c4），值是对应分辨率的 FPN 输出
        return results
