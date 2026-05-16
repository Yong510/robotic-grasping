import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

sys.path.insert(0, '/root/autodl-tmp/robotic-grasping')

from inference.models import get_network
from utils.dataset_processing.jacquard import JacquardDataset
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import save_results

# 配置参数
MODEL_PATH = 'logs/260417_2107_MFENet_Resume_Final/epoch_09_iou_0.00'
DATASET_PATH = 'data/jacquard/Jacquard_Dataset_0'
INPUT_SIZE = 300
NUM_VIS = 10
OUTPUT_DIR = 'results/inference_vis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型
net = get_network('mfenet')
checkpoint = torch.load(MODEL_PATH, map_location=device)

# 处理不同保存格式
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
else:
    net = checkpoint  # 直接整个模型对象

net.to(device)
net.eval()

# 加载数据集
dataset = JacquardDataset(DATASET_PATH, output_size=INPUT_SIZE,
                          include_depth=True, include_rgb=True)
indices = list(range(len(dataset)))
split = int(0.9 * len(dataset))
val_indices = indices[split:][:NUM_VIS]

print(f"Processing {len(val_indices)} validation samples...")

for i, val_idx in enumerate(val_indices):
    # 获取原始RGB图像（用于可视化）
    rgb_img = dataset.get_rgb(val_idx, rot=0, zoom=1.0, normalise=False)
    depth_img = dataset.get_depth(val_idx, rot=0, zoom=1.0)

    # 构建模型输入（已经过预处理）
    x, _, _, _, _ = dataset[val_idx]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pos, cos, sin, width = net(x)

    # 后处理得到抓取质量图、角度图和宽度图
    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)

    # 调用原项目自带的保存函数（生成带抓取框的图片）
    save_results(rgb_img, depth_img, q_img, ang_img, no_grasps=1,
                 grasp_width_img=width_img, save_dir=OUTPUT_DIR, idx=i)

print(f"Inference completed. Images saved to {OUTPUT_DIR}")
