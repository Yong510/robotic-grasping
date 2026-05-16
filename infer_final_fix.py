import os
import sys
import glob
import torch
import numpy as np
import cv2

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
    # 尝试用通配符找任意 .tiff 文件
    tiff_files = glob.glob(base + '*.tiff')
    for tf in tiff_files:
        # 优先 perfect，其次 stereo
        if 'perfect' in tf or 'stereo' in tf:
            return tf
    return None

# 切换到输出目录，让 save_results 存图在此
orig_dir = os.getcwd()
os.chdir(OUTPUT_DIR)

generated = 0
idx = 0
while generated < NUM_VIS and idx < len(rgb_files):
    rgb_path = rgb_files[idx]
    idx += 1

    depth_path = find_depth_file(rgb_path)
    if depth_path is None:
        print(f"Skipping {os.path.basename(rgb_path)}: no depth file")
        continue

    x = preprocess(rgb_path, depth_path).to(device)
    with torch.no_grad():
        pos, cos, sin, width = net(x)

    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)

    rgb_vis = img_utils.Image.from_file(rgb_path)
    depth_vis = img_utils.Image.from_file(depth_path)

    # 生成完整多子图
    save_results(rgb_vis.img, q_img, ang_img, depth_img=depth_vis.img,
                 no_grasps=1, grasp_width_img=width_img)

    # 重命名
    default_name = f"{generated}_grasp.png"
    new_name = f"sample_{generated:02d}.png"
    if os.path.exists(default_name):
        os.rename(default_name, new_name)
        print(f"Saved {new_name} from {os.path.basename(rgb_path)}")
        generated += 1
    else:
        print(f"Warning: save_results did not create {default_name}")

os.chdir(orig_dir)
print(f"Done. Generated {generated} images in {OUTPUT_DIR}")
