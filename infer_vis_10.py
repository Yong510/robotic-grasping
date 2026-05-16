import os
import glob
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing.grasp import detect_grasps


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 10 inference samples")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weight")
    parser.add_argument("--dataset", type=str, default="jacquard")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--input-size", type=int, default=300)
    parser.add_argument("--use-depth", type=int, default=1)
    parser.add_argument("--use-rgb", type=int, default=1)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="vis_epoch48_10")
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args()


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    # x: [C, H, W]
    c = x.shape[0]
    arr = x.detach().cpu().numpy()

    if c >= 4:
        rgb = np.transpose(arr[1:4], (1, 2, 0))
    elif c == 3:
        rgb = np.transpose(arr, (1, 2, 0))
    else:
        gray = arr[0]
        rgb = np.stack([gray, gray, gray], axis=-1)

    return normalize_for_display(rgb)


def get_prediction(model, xc):
    with torch.no_grad():
        out = model(xc)
        if isinstance(out, dict):
            pos = out["pos"]
            cos = out["cos"]
            sin = out["sin"]
            width = out["width"]
        else:
            pos, cos, sin, width = out
    return pos, cos, sin, width


def draw_grasp(ax, grasp):
    # 优先调用仓库里 grasp 对象的 plot 方法
    try:
        grasp.plot(ax)
        return
    except Exception:
        pass

    # 退化方案：只画中心点
    try:
        cy, cx = grasp.center
        ax.scatter([cx], [cy], s=40)
    except Exception:
        pass


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = get_device(args.cpu)

    print(f"Loading model from: {args.weights}")
    model = torch.load(args.weights, map_location=device)
    model = model.to(device)
    model.eval()

    print("Loading dataset...")
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=0.0,
        random_rotate=False,
        random_zoom=False,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb
    )

    n = len(dataset)
    num = min(args.num_images, n)
    indices = np.linspace(0, n - 1, num=num, dtype=int)

    # 总览图：只放 RGB + 抓取框
    cols = 5
    rows = math.ceil(num / cols)
    fig_overview, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for plot_i, idx in enumerate(indices):
        sample = dataset[idx]
        x, y, didx, rot, zoom = sample

        xc = x.unsqueeze(0).to(device)
        pos, cos, sin, width = get_prediction(model, xc)

        q_img, ang_img, width_img = post_process_output(pos, cos, sin, width)
        grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=1)

        rgb = tensor_to_rgb(x)

        # 详细图
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].imshow(rgb)
        if len(grasps) > 0:
            draw_grasp(axs[0, 0], grasps[0])
        axs[0, 0].set_title(f"RGB + Predicted Grasp (idx={idx})")
        axs[0, 0].axis("off")

        im1 = axs[0, 1].imshow(q_img)
        axs[0, 1].set_title("Grasp Quality")
        axs[0, 1].axis("off")
        fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        im2 = axs[1, 0].imshow(ang_img, cmap="twilight")
        axs[1, 0].set_title("Grasp Angle")
        axs[1, 0].axis("off")
        fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

        im3 = axs[1, 1].imshow(width_img)
        axs[1, 1].set_title("Grasp Width")
        axs[1, 1].axis("off")
        fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        save_path = os.path.join(args.save_dir, f"sample_{plot_i:02d}_idx_{idx}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # 总览图
        r, c = divmod(plot_i, cols)
        axes[r, c].imshow(rgb)
        if len(grasps) > 0:
            draw_grasp(axes[r, c], grasps[0])
        axes[r, c].set_title(f"idx={idx}")
        axes[r, c].axis("off")

        print(f"[{plot_i+1}/{num}] saved: {save_path}")

    # 去掉多余子图
    for k in range(num, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig_overview.tight_layout()
    overview_path = os.path.join(args.save_dir, "overview_10_samples.png")
    fig_overview.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close(fig_overview)

    print(f"Done. Overview saved to: {overview_path}")


if __name__ == "__main__":
    main()
