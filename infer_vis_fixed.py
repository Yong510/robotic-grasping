import sys
import torch
import os

sys.path.insert(0, '/root/autodl-tmp/robotic-grasping')

from inference.models import get_network
from utils.data.jacquard_data import JacquardDataset
from inference.post_process import post_process_output
from utils.visualisation.plot import save_results

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
if isinstance(checkpoint, dict):
    net.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
else:
    net = checkpoint
net.to(device)
net.eval()

# 加载数据集
dataset = JacquardDataset(DATASET_PATH, output_size=INPUT_SIZE,
                          include_depth=True, include_rgb=True)
indices = list(range(len(dataset)))
split = int(0.9 * len(dataset))
val_indices = indices[split:][:NUM_VIS]

print(f"Processing {len(val_indices)} validation samples...")

# 保存当前工作目录，进入输出目录（让 save_results 把图存在这里）
orig_dir = os.getcwd()
os.chdir(OUTPUT_DIR)

for i, val_idx in enumerate(val_indices):
    rgb_img = dataset.get_rgb(val_idx, rot=0, zoom=1.0, normalise=False)
    depth_img = dataset.get_depth(val_idx, rot=0, zoom=1.0)

    x, _, _, _, _ = dataset[val_idx]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pos, cos, sin, width = net(x)

    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)

    # save_results 会自动生成文件名，如 0_grasp.png, 1_grasp.png 等
    save_results(rgb_img, q_img, ang_img, depth_img=depth_img,
                 no_grasps=1, grasp_width_img=width_img)

# 恢复工作目录
os.chdir(orig_dir)

print(f"Inference completed. Images saved to {OUTPUT_DIR}")
