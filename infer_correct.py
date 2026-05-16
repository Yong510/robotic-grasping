import os
import sys
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/autodl-tmp/robotic-grasping')

from inference.models import get_network
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing import image as img_utils

MODEL_PATH = 'logs/260417_2107_MFENet_Resume_Final/epoch_09_iou_0.00'
DATA_ROOT = 'data/jacquard/Jacquard_Dataset_0'
INPUT_SIZE = 300
OUTPUT_DIR = 'results/inference_vis'
NUM_VIS = 10  # 只可视化前10个样本

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

# 收集所有RGB文件作为样本列表
rgb_files = sorted(glob.glob(os.path.join(DATA_ROOT, '*', '*_RGB.png')))
print(f"Total RGB files found: {len(rgb_files)}")

# 只取验证集部分（后10%）或直接取前NUM_VIS个
# 这里为了快速可视化，我们直接取前 NUM_VIS 个
sample_paths = rgb_files[:NUM_VIS]
print(f"Will visualize {len(sample_paths)} samples.")

def preprocess(rgb_path, depth_path):
    """将RGB和深度图拼接为4通道输入张量"""
    rgb = img_utils.Image.from_file(rgb_path)
    depth = img_utils.Image.from_file(depth_path)

    # 深度图可能为3维，取第一个通道
    if depth.img.ndim == 3:
        depth_data = depth.img[:, :, 0]
    else:
        depth_data = depth.img

    # 归一化深度图：减去均值，clip到[-1,1]
    depth_data = depth_data.astype(np.float32)
    depth_norm = np.clip((depth_data - depth_data.mean()), -1, 1)

    # RGB归一化到[0,1]
    rgb_data = rgb.img.astype(np.float32) / 255.0

    # 尺寸对齐（深度图可能与RGB分辨率不同）
    if rgb_data.shape[:2] != depth_norm.shape[:2]:
        depth_norm = cv2.resize(depth_norm, (rgb_data.shape[1], rgb_data.shape[0]))

    # 拼接：RGB (H,W,3) + Depth (H,W,1) -> (H,W,4)
    depth_norm = depth_norm[:, :, np.newaxis]
    x = np.concatenate((rgb_data, depth_norm), axis=2)

    # 缩放到网络输入尺寸
    x = cv2.resize(x, (INPUT_SIZE, INPUT_SIZE))

    # 转换为张量 (C,H,W) 并增加batch维度
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
    return x

def find_depth_file(rgb_path):
    """根据RGB文件名查找对应的深度图（优先perfect，否则stereo）"""
    base = rgb_path.replace('_RGB.png', '')
    # 尝试 perfect_depth.tiff
    depth_path = base + '_perfect_depth.tiff'
    if os.path.exists(depth_path):
        return depth_path
    # 尝试 stereo_depth.tiff
    depth_path = base + '_stereo_depth.tiff'
    if os.path.exists(depth_path):
        return depth_path
    return None

for i, rgb_path in enumerate(sample_paths):
    depth_path = find_depth_file(rgb_path)
    if depth_path is None:
        print(f"Skipping {rgb_path}: no depth file found")
        continue

    x = preprocess(rgb_path, depth_path).to(device)

    with torch.no_grad():
        pos, cos, sin, width = net(x)

    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

    # 可视化
    rgb_vis = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb_vis)
    ax1.set_title('RGB')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(rgb_vis)
    if grasps:
        g = grasps[0]
        center = (g.center[1], g.center[0])
        length = g.length
        width_g = g.width
        angle = g.angle
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (center[0] - length/2, center[1] - width_g/2),
            length, width_g,
            angle=np.rad2deg(angle),
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.scatter(center[0], center[1], c='red', s=20)
    ax2.set_title('Predicted Grasp')
    ax2.axis('off')

    save_path = os.path.join(OUTPUT_DIR, f'grasp_{i:02d}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

print(f"Inference completed. {len(sample_paths)} images saved to {OUTPUT_DIR}")
