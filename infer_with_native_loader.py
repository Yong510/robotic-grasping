import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, '/root/autodl-tmp/robotic-grasping')

from inference.models import get_network
from inference.post_process import post_process_output
from utils.data.jacquard_data import JacquardDataset
from utils.dataset_processing.grasp import detect_grasps

# 配置
MODEL_PATH = 'logs/260417_2332_MFENet_50epochs_1k_Final/epoch_28_iou_0.00'
DATASET_PATH = 'data/jacquard/Jacquard_Dataset_0'
INPUT_SIZE = 300
OUTPUT_DIR = 'results/native_vis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
net = get_network('mfenet')
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict):
    net.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
else:
    net = checkpoint
net.to(device)
net.eval()

# 使用原版数据集类
dataset = JacquardDataset(DATASET_PATH, output_size=INPUT_SIZE,
                          include_depth=True, include_rgb=True)

# 取验证集部分（后10%）
indices = list(range(len(dataset)))
split = int(0.9 * len(dataset))
val_indices = indices[split:]

# 只可视化第一个样本
idx = val_indices[0]
x, _, _, _, _ = dataset[idx]
x = x.unsqueeze(0).to(device)

with torch.no_grad():
    pos, cos, sin, width = net(x)

q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)
grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

# 获取原始图像用于显示
rgb_img = dataset.get_rgb(idx, rot=0, zoom=1.0, normalise=False)
depth_img = dataset.get_depth(idx, rot=0, zoom=1.0)

# 绘制四子图
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. RGB + 抓取框
axes[0, 0].imshow(rgb_img)
if grasps:
    g = grasps[0]
    center = (g.center[1], g.center[0])
    length = g.length
    width_g = g.width
    angle = np.rad2deg(g.angle)
    rect = plt.Rectangle((center[0]-length/2, center[1]-width_g/2),
                         length, width_g, angle=angle,
                         linewidth=2, edgecolor='lime', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].scatter(center[0], center[1], c='red', s=20)
axes[0, 0].set_title('RGB with Predicted Grasp')
axes[0, 0].axis('off')

# 2. 抓取质量图
im2 = axes[0, 1].imshow(q_img, cmap='jet', vmin=0, vmax=1)
axes[0, 1].set_title('Grasp Quality (pos)')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

# 3. 角度图
ang_display = (ang_img + np.pi/2) / np.pi
axes[1, 0].imshow(ang_display, cmap='hsv')
axes[1, 0].set_title('Grasp Angle')
axes[1, 0].axis('off')

# 4. 宽度图
im4 = axes[1, 1].imshow(width_img, cmap='plasma')
axes[1, 1].set_title('Grasp Width')
axes[1, 1].axis('off')
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'native_sample.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图片已保存至 {save_path}")
