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
from utils.dataset_processing import image as img_utils
from utils.visualisation.plot import save_results

MODEL_PATH = 'logs/260417_2107_MFENet_Resume_Final/epoch_09_iou_0.00'
DATA_ROOT = 'data/jacquard/Jacquard_Dataset_0'
INPUT_SIZE = 300
OUTPUT_DIR = 'results/full_vis'
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

# 收集所有RGB文件
rgb_files = sorted(glob.glob(os.path.join(DATA_ROOT, '*', '*_RGB.png')))
print(f"Total RGB files found: {len(rgb_files)}")
sample_paths = rgb_files[:NUM_VIS]

def preprocess(rgb_path, depth_path):
    rgb = img_utils.Image.from_file(rgb_path)
    depth = img_utils.Image.from_file(depth_path)
    if depth.img.ndim == 3:
        depth_data = depth.img[:, :, 0]
    else:
        depth_data = depth.img
    depth_data = depth_data.astype(np.float32)
    depth_norm = np.clip((depth_data - depth_data.mean()), -1, 1)
    rgb_data = rgb.img.astype(np.float32) / 255.0
    if rgb_data.shape[:2] != depth_norm.shape[:2]:
        depth_norm = cv2.resize(depth_norm, (rgb_data.shape[1], rgb_data.shape[0]))
    depth_norm = depth_norm[:, :, np.newaxis]
    x = np.concatenate((rgb_data, depth_norm), axis=2)
    x = cv2.resize(x, (INPUT_SIZE, INPUT_SIZE))
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
    return x

def find_depth_file(rgb_path):
    base = rgb_path.replace('_RGB.png', '')
    depth_path = base + '_perfect_depth.tiff'
    if os.path.exists(depth_path):
        return depth_path
    depth_path = base + '_stereo_depth.tiff'
    if os.path.exists(depth_path):
        return depth_path
    return None

# 切换到输出目录，让 save_results 把图片存到这里
orig_dir = os.getcwd()
os.chdir(OUTPUT_DIR)

for i, rgb_path in enumerate(sample_paths):
    depth_path = find_depth_file(rgb_path)
    if depth_path is None:
        print(f"Skipping {rgb_path}: no depth file")
        continue

    x = preprocess(rgb_path, depth_path).to(device)

    with torch.no_grad():
        pos, cos, sin, width = net(x)

    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)

    # 获取原始RGB图像用于背景绘制
    rgb_vis = img_utils.Image.from_file(rgb_path)
    depth_vis = img_utils.Image.from_file(depth_path)

    # 调用原版 save_results 生成包含 pos/ang/width 的完整多子图
    # 注意参数顺序：rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None
    save_results(rgb_vis.img, q_img, ang_img, depth_img=depth_vis.img,
                 no_grasps=1, grasp_width_img=width_img)

    # 因为 save_results 生成的文件名类似 0_grasp.png，我们手动重命名加上前缀以区分样本
    default_name = f"{i}_grasp.png"
    if os.path.exists(default_name):
        new_name = f"sample_{i:02d}.png"
        os.rename(default_name, new_name)
        print(f"Saved {new_name}")
    else:
        print(f"Warning: {default_name} not created for sample {i}")

os.chdir(orig_dir)
print(f"Done. Images saved in {OUTPUT_DIR}")
