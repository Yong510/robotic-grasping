import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader

from utils.data.ocid_grasp_object_data import OCIDGraspObjectDataset


def main():
    root = "/root/autodl-tmp/datasets/OCID_grasp/OCID_grasp"
    if not os.path.exists(root):
        root = os.path.join(PROJECT_ROOT, "data", "OCID_grasp", "OCID_grasp")

    print("[INFO] OCID root:", root)

    ds = OCIDGraspObjectDataset(
        root=root,
        split="train",
        output_size=300,
        max_objects=20,
        include_depth=True,
        include_rgb=True,
        max_grasps=300,
    )

    print("[INFO] dataset length:", len(ds))
    sample = ds[0]
    print("[INFO] sample meta:", sample["meta"])
    print("[INFO] input:", tuple(sample["input"].shape), sample["input"].dtype)
    print("[INFO] pos:", tuple(sample["pos"].shape), sample["pos"].min().item(), sample["pos"].max().item())
    print("[INFO] cos:", tuple(sample["cos"].shape))
    print("[INFO] sin:", tuple(sample["sin"].shape))
    print("[INFO] width:", tuple(sample["width"].shape), sample["width"].min().item(), sample["width"].max().item())
    print("[INFO] masks:", tuple(sample["masks"].shape), sample["masks"].sum().item())
    print("[INFO] object_labels:", sample["object_labels"][:20].tolist())

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("[OK] dataloader batch loaded")
    print("[INFO] batch input:", tuple(batch["input"].shape))
    print("[INFO] batch masks:", tuple(batch["masks"].shape))
    print("[INFO] batch labels:", tuple(batch["object_labels"].shape))
    print("[OK] OCID dataset check passed")


if __name__ == "__main__":
    main()
