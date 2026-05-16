import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OCIDGraspObjectDataset(Dataset):
    """
    Object-centric OCID-Grasp dataset adapter for LG-Risk-MFENet.

    Expected OCID-Grasp structure:
        root/
          data_split/training_0.txt
          data_split/validation_0.txt
          ARID20/table/top/seq01/
            rgb/*.png
            depth/*.png
            label/*.png                      # instance ids: 0 background, 1..N objects
            seg_mask_labeled_combi/*.png     # semantic class ids: 0..31
            Annotations/*.txt                # grasp rectangles, 4 lines per grasp

    Each sample returns:
        input:         [4, H, W]
        pos/cos/sin/width: [1, H, W]
        masks:         [max_objects, H, W]
        object_labels: [max_objects], invalid = -1
    """

    def __init__(
        self,
        root,
        split="train",
        output_size=300,
        max_objects=20,
        include_depth=True,
        include_rgb=True,
        max_grasps=None,
    ):
        self.root = Path(root)
        self.split = split
        self.output_size = int(output_size)
        self.max_objects = int(max_objects)
        self.include_depth = bool(include_depth)
        self.include_rgb = bool(include_rgb)
        self.max_grasps = max_grasps

        if not self.root.exists():
            raise FileNotFoundError(f"OCID-Grasp root not found: {self.root}")

        if split in ["train", "training"]:
            split_file = self.root / "data_split" / "training_0.txt"
        elif split in ["val", "valid", "validation"]:
            split_file = self.root / "data_split" / "validation_0.txt"
        else:
            raise ValueError(f"Unknown split={split}. Use train or val.")

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seq_rel, img_name = line.split(",")
                self.samples.append((seq_rel, img_name))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {split_file}")

    def __len__(self):
        return len(self.samples)

    def _paths(self, seq_rel, img_name):
        seq_dir = self.root / seq_rel
        stem = Path(img_name).stem
        return {
            "rgb": seq_dir / "rgb" / img_name,
            "depth": seq_dir / "depth" / img_name,
            "inst": seq_dir / "label" / img_name,
            "sem": seq_dir / "seg_mask_labeled_combi" / img_name,
            "ann": seq_dir / "Annotations" / f"{stem}.txt",
        }

    @staticmethod
    def _safe_read_image(path, flags):
        img = cv2.imread(str(path), flags)
        if img is None:
            raise FileNotFoundError(str(path))
        return img

    @staticmethod
    def _normalize_depth(depth):
        depth = depth.astype(np.float32)
        valid = depth > 0
        if valid.any():
            mean = depth[valid].mean()
            std = depth[valid].std() + 1e-6
            depth = (depth - mean) / std
            depth[~valid] = 0.0
        return depth

    def _load_rgbd(self, rgb_path, depth_path):
        channels = []

        if self.include_rgb:
            rgb = self._safe_read_image(rgb_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb.astype(np.float32) / 255.0
            channels.append(rgb.transpose(2, 0, 1))

        if self.include_depth:
            depth = self._safe_read_image(depth_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(depth, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            depth = self._normalize_depth(depth)
            channels.append(depth[None, ...].astype(np.float32))

        if not channels:
            raise ValueError("At least one of include_rgb/include_depth must be True.")

        return np.concatenate(channels, axis=0).astype(np.float32)

    def _load_object_masks_and_labels(self, inst_path, sem_path):
        inst = self._safe_read_image(inst_path, cv2.IMREAD_UNCHANGED)
        sem = self._safe_read_image(sem_path, cv2.IMREAD_UNCHANGED)

        inst_resized = cv2.resize(inst, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        sem_resized = cv2.resize(sem, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        instance_ids = [int(v) for v in np.unique(inst_resized) if int(v) != 0]
        instance_ids = instance_ids[: self.max_objects]

        masks = np.zeros((self.max_objects, self.output_size, self.output_size), dtype=np.float32)
        object_labels = np.full((self.max_objects,), fill_value=-1, dtype=np.int64)

        for obj_idx, inst_id in enumerate(instance_ids):
            m = inst_resized == inst_id
            if not m.any():
                continue
            masks[obj_idx] = m.astype(np.float32)

            sem_values = sem_resized[m]
            sem_values = sem_values[sem_values > 0]
            if sem_values.size == 0:
                label = 31  # unknown
            else:
                vals, counts = np.unique(sem_values.astype(np.int64), return_counts=True)
                label = int(vals[np.argmax(counts)])
            object_labels[obj_idx] = label

        return masks, object_labels

    @staticmethod
    def _parse_grasp_rectangles(annotation_path):
        if not annotation_path.exists():
            return []

        pts = []
        with open(annotation_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                pts.append([float(parts[0]), float(parts[1])])

        rects = []
        for i in range(0, len(pts) - 3, 4):
            rect = np.asarray(pts[i : i + 4], dtype=np.float32)
            if rect.shape == (4, 2):
                rects.append(rect)
        return rects

    def _generate_grasp_maps(self, annotation_path, orig_h=480, orig_w=640):
        h = w = self.output_size
        pos = np.zeros((h, w), dtype=np.float32)
        cos = np.zeros((h, w), dtype=np.float32)
        sin = np.zeros((h, w), dtype=np.float32)
        width = np.zeros((h, w), dtype=np.float32)

        rects = self._parse_grasp_rectangles(annotation_path)
        if self.max_grasps is not None:
            rects = rects[: self.max_grasps]

        sx = w / float(orig_w)
        sy = h / float(orig_h)
        scale_width = (sx + sy) * 0.5

        for rect in rects:
            rect_scaled = rect.copy()
            rect_scaled[:, 0] *= sx
            rect_scaled[:, 1] *= sy

            center = rect_scaled.mean(axis=0)
            edge01 = rect_scaled[1] - rect_scaled[0]
            edge12 = rect_scaled[2] - rect_scaled[1]
            len01 = float(np.linalg.norm(edge01))
            len12 = float(np.linalg.norm(edge12))

            if len01 < 1e-3 or len12 < 1e-3:
                continue

            if len01 >= len12:
                angle = math.atan2(edge01[1], edge01[0])
                grasp_width = len12 / max(scale_width, 1e-6)
            else:
                angle = math.atan2(edge12[1], edge12[0])
                grasp_width = len01 / max(scale_width, 1e-6)

            poly = np.round(rect_scaled).astype(np.int32)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

            cv2.fillConvexPoly(pos, poly, 1.0)
            tmp = np.zeros_like(pos)
            cv2.fillConvexPoly(tmp, poly, 1.0)
            mask = tmp > 0
            cos[mask] = math.cos(2.0 * angle)
            sin[mask] = math.sin(2.0 * angle)
            width[mask] = min(grasp_width, 150.0) / 150.0

        return pos[None, ...], cos[None, ...], sin[None, ...], width[None, ...]

    def __getitem__(self, idx):
        seq_rel, img_name = self.samples[idx]
        paths = self._paths(seq_rel, img_name)

        x = self._load_rgbd(paths["rgb"], paths["depth"])
        masks, object_labels = self._load_object_masks_and_labels(paths["inst"], paths["sem"])
        pos, cos, sin, width = self._generate_grasp_maps(paths["ann"])

        return {
            "input": torch.from_numpy(x).float(),
            "pos": torch.from_numpy(pos).float(),
            "cos": torch.from_numpy(cos).float(),
            "sin": torch.from_numpy(sin).float(),
            "width": torch.from_numpy(width).float(),
            "masks": torch.from_numpy(masks).float(),
            "object_labels": torch.from_numpy(object_labels).long(),
            "meta": {
                "seq": seq_rel,
                "image": img_name,
            },
        }
