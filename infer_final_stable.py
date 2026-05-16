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

# ========== 配置参数 ==========
MODEL_PATH = 'logs/260417_2107_MFENet_Resume_Final/epoch_09_iou_0.00'
DATA_ROOT = 'data/jacquard/Jacquard_Dataset_0'
INPUT_SIZE = 300
OUTPUT_DIR = 'results/final_vis'
NUM_VIS = 10
# =============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 加载模型 ----------
net = get_network('mfenet')
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict):
    net.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
else:
    net = checkpoint
net.to(device)
net.eval()

# ---------- 扫描所有RGB文件 ----------
rgb_files = sorted(glob.glob(os.path.join(DATA_ROOT, '*', '*_RGB.png')))
print(f"找到 {len(rgb_files)} 个RGB文件")

def find_depth(rgb_path):
    """根据RGB路径查找对应的深度图（优先perfect，否则stereo）"""
    dir_name = os.path.dirname(rgb_path)
    base_name = os.path.basename(rgb_path).replace('_RGB.png', '')
    # 尝试 perfect_depth.tiff
    perfect = os.path.join(dir_name, base_name + '_perfect_depth.tiff')
    if os.path.exists(perfect):
        return perfect
    # 尝试 stereo_depth.tiff
    stereo = os.path.join(dir_name, base_name + '_stereo_depth.tiff')
    if os.path.exists(stereo):
        return stereo
    return None

def preprocess(rgb_path, depth_path):
    """预处理：RGB+Depth -> 4通道张量 (1,4,H,W)"""
    rgb = img_utils.Image.from_file(rgb_path)
    depth = img_utils.Image.from_file(depth_path)

    # 处理深度图维度
    if depth.img.ndim == 3:
        depth_data = depth.img[:, :, 0]
    else:
        depth_data = depth.img

    # 归一化深度图（与训练一致）
    depth_data = depth_data.astype(np.float32)
    depth_norm = np.clip((depth_data - depth_data.mean()), -1, 1)

    # 归一化RGB
    rgb_data = rgb.img.astype(np.float32) / 255.0

    # 对齐尺寸
    if rgb_data.shape[:2] != depth_norm.shape[:2]:
        depth_norm = cv2.resize(depth_norm, (rgb_data.shape[1], rgb_data.shape[0]))

    # 拼接为4通道
    depth_norm = depth_norm[:, :, np.newaxis]
    x = np.concatenate((rgb_data, depth_norm), axis=2)

    # 缩放至网络输入大小
    x = cv2.resize(x, (INPUT_SIZE, INPUT_SIZE))

    # 转换为张量 (C,H,W) 并增加batch维度
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
    return x

def draw_grasp(ax, grasp, rgb_shape):
    """在给定的ax上绘制抓取矩形"""
    if grasp is None:
        return
    center = (grasp.center[1], grasp.center[0])  # (x, y)
    length = grasp.length
    width = grasp.width
    angle = np.rad2deg(grasp.angle)
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (center[0] - length/2, center[1] - width/2),
        length, width, angle=angle,
        linewidth=2, edgecolor='lime', facecolor='none'
    )
    ax.add_patch(rect)
    ax.scatter(center[0], center[1], c='red', s=20)

# ---------- 生成可视化 ----------
generated = 0
idx = 0
while generated < NUM_VIS and idx < len(rgb_files):
    rgb_path = rgb_files[idx]
    idx += 1

    depth_path = find_depth(rgb_path)
    if depth_path is None:
        print(f"跳过 {os.path.basename(rgb_path)}：无对应深度图")
        continue

    # 预处理
    x = preprocess(rgb_path, depth_path).to(device)

    # 推理
    with torch.no_grad():
        pos, cos, sin, width = net(x)

    # 后处理得到抓取图
    q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)

    # 检测最佳抓取
    from utils.dataset_processing.grasp import detect_grasps
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    best_grasp = grasps[0] if grasps else None

    # 读取原始图像用于绘图
    rgb_vis = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth_vis = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 创建子图 (2行2列)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1. RGB + 抓取框
    axes[0, 0].imshow(rgb_vis)
    draw_grasp(axes[0, 0], best_grasp, rgb_vis.shape)
    axes[0, 0].set_title('RGB with Predicted Grasp')
    axes[0, 0].axis('off')

    # 2. 抓取质量热力图 (pos)
    im2 = axes[0, 1].imshow(q_img, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Grasp Quality (pos)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 3. 角度图 (将ang_img映射到HSV显示)
    # ang_img 的值范围是 [-pi/2, pi/2]，我们将其映射到 [0,1] 然后用 HSV 显示
    ang_display = (ang_img + np.pi/2) / np.pi  # 归一化到 [0,1]
    axes[1, 0].imshow(ang_display, cmap='hsv')
    axes[1, 0].set_title('Grasp Angle')
    axes[1, 0].axis('off')

    # 4. 宽度图
    im4 = axes[1, 1].imshow(width_img, cmap='plasma')
    axes[1, 1].set_title('Grasp Width')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'sample_{generated:02d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存 {save_path}  (来源: {os.path.basename(rgb_path)})")
    generated += 1

print(f"\n✅ 完成！共生成 {generated} 张图片，保存在 {OUTPUT_DIR}")
