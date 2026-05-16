import random
import numpy as np
import torch
from torch.utils.data import Dataset


class GraspDatasetBase(Dataset):
    """
    Base class for grasp datasets. Handles data augmentation and input building.
    """
    def __init__(self, output_size=300, include_depth=True, include_rgb=True, random_rotate=False,
                 random_zoom=False, input_only=False):
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only

        # 子类需定义以下属性
        self.length = 0
        self.grasp_files = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rot = 0.0
        zoom = 1.0
        if self.random_rotate:
            rot = random.uniform(-np.pi, np.pi)
        if self.random_zoom:
            zoom = random.uniform(0.8, 1.2)

        # 加载深度图（子类实现）
        depth_img = None
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom)

        # 加载 RGB 图（子类实现）
        rgb_img = None
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom)

        # 构建网络输入：拼接为 (H, W, 4)
        x = self._build_input(rgb_img, depth_img)

        # 获取抓取真值并转换为图像
        bbs = self.get_gtbb(idx, rot, zoom)
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        pos_img = np.expand_dims(pos_img, axis=0)
        # 处理角度图维度（可能为2D或3D）
        if ang_img.ndim == 2:
            ang_img = np.stack([np.cos(ang_img), np.sin(ang_img)], axis=-1)
        cos_img = np.expand_dims(ang_img[:, :, 0], axis=0)
        sin_img = np.expand_dims(ang_img[:, :, 1], axis=0)
        sin_img = np.expand_dims(ang_img[:, :, 1], axis=0)
        width_img = np.expand_dims(width_img, axis=0)

        # 转换为 PyTorch 张量
        x = torch.from_numpy(x).float()
        # 转换为 (C, H, W) 格式
        x = x.permute(2, 0, 1)
        pos = torch.from_numpy(pos_img).float()
        cos = torch.from_numpy(cos_img).float()
        sin = torch.from_numpy(sin_img).float()
        width = torch.from_numpy(width_img).float()

        return x, (pos, cos, sin, width), idx, rot, zoom

    def _build_input(self, rgb_img, depth_img):
        """
        将 RGB 和深度图拼接为 (H, W, 4) 数组。
        处理各种维度异常（如通道在前、单通道扩展等）。
        """
        channels = []
        if self.include_rgb and rgb_img is not None:
            # 确保 RGB 为 (H, W, 3)，值域 [0, 1]
            if rgb_img.ndim == 4:
                rgb_img = rgb_img.squeeze(0)
            if rgb_img.shape[-1] != 3 and rgb_img.shape[0] == 3:
                rgb_img = np.transpose(rgb_img, (1, 2, 0))
            if rgb_img.max() > 1.0:
                rgb_img = rgb_img.astype(np.float32) / 255.0
            channels.append(rgb_img)

        if self.include_depth and depth_img is not None:
            # 确保深度图为 (H, W, 1)
            if depth_img.ndim == 3:
                if depth_img.shape[-1] == 3:  # 若深度图被误存为3通道
                    depth_img = depth_img[:, :, 0]
                elif depth_img.shape[0] == 1:
                    depth_img = depth_img.squeeze(0)
            depth_img = depth_img[:, :, np.newaxis]
            if depth_img.max() > 1.0:
                depth_img = depth_img.astype(np.float32) / 255.0
            channels.append(depth_img)

        if len(channels) == 2:
            x = np.concatenate(channels, axis=2)
        elif len(channels) == 1:
            x = channels[0]
        else:
            raise ValueError("At least one modality (RGB or Depth) must be included.")
        return x

    # 以下方法需由子类实现
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        raise NotImplementedError()
