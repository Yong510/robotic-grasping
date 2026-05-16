import os
import sys
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, '/root/autodl-tmp/robotic-grasping')

from inference.models import get_network
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing import image as img_utils

MODEL_PATH = 'logs/260417_2107_MFENet_Resume_Final/epoch_09_iou_0.00'
DATA_ROOT = 'data/jacquard/Jacquard_Dataset_0'
INPUT_SIZE = 300
OUTPUT_DIR = 'results/inference_vis'
NUM_VIS = 10

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

# 收集所有场景目录
scene_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, '*')))
split_idx = int(0.9 * len(scene_dirs))
val_scenes = scene_dirs[split_idx:split_idx + NUM_VIS]

print(f"Found {len(scene_dirs)} scenes, using {len(val_scenes)} for visualization.")

def preprocess(rgb_path, depth_path):
    """将RGB和深度图拼接为4通道输入，并resize到INPUT_SIZE"""
    rgb = img_utils.Image.from_file(rgb_path)
    depth = img_utils.Image.from_file(depth_path)
    # 深度图归一化
    depth_img = depth.img.astype(np.float32)
    depth_img = np.clip((depth_img - depth_img.mean()), -1, 1)
    # RGB归一化
    rgb_img = rgb.img.astype(np.float32) / 255.0
    # 拼接
    x = np.concatenate((rgb_img, np.expand_dims(depth_img, axis=2)), axis=2)
    x = cv2.resize(x, (INPUT_SIZE, INPUT_SIZE))
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
    return x

for i, scene in enumerate(val_scenes):
    # 查找文件
    rgb_files = glob.glob(os.path.join(scene, '*_RGB.png'))
    depth_files = glob.glob(os.path.join(scene, '*_perfect_depth.tiff'))
    if not rgb_files or not depth_files:
        print(f"Skipping {scene}: missing files")
        continue

    rgb_path = rgb_files[0]
    depth_path = depth_files[0]

    # 预处理
    x = preprocess(rgb_path, depth_path).to(device)

    with torch.no_grad():
        pos, cos, sin, width = net(x)

    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

    # 可视化
    rgb_vis = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth_vis = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb_vis)
    ax1.set_title('RGB')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(rgb_vis)
    # 绘制抓取框（若存在）
    if grasps:
        g = grasps[0]
        center = (g.center[1], g.center[0])
        angle = g.angle
        width = g.length
        height = g.width
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (center[0] - width/2, center[1] - height/2),
            width, height,
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

print(f"Inference completed. {len(val_scenes)} images saved to {OUTPUT_DIR}")
