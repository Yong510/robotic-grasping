import argparse
import datetime
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import tensorboardX
except Exception:
    tensorboardX = None

from inference.models.mfenet import MFENet
from inference.models.object_semantic_mfenet import ObjectSemanticMFENet
from losses.object_semantic_grasp_loss import ObjectSemanticGraspLoss
from utils.data.ocid_grasp_object_data import OCIDGraspObjectDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train LG-Risk-MFENet on OCID-Grasp")

    parser.add_argument("--dataset-root", type=str,
                        default="/root/autodl-tmp/datasets/OCID_grasp/OCID_grasp")
    parser.add_argument("--save-dir", type=str, default="logs/ocid_object_semantic_mfenet")
    parser.add_argument("--description", type=str, default="ocid_lg_risk_mfenet")

    parser.add_argument("--input-size", type=int, default=300)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument("--max-grasps", type=int, default=500)
    parser.add_argument("--num-classes", type=int, default=32,
                        help="OCID labels include background=0 and unknown=31, so use 32 classes.")
    parser.add_argument("--num-risk-levels", type=int, default=3)
    parser.add_argument("--feature-key", type=str, default="p2",
                        choices=["p2", "p3", "p4", "c2", "c3", "c4", "grasp_feat"])

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batches-per-epoch", type=int, default=0,
                        help="0 means use the whole dataloader; positive value limits batches per epoch for debug.")
    parser.add_argument("--lr-base", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cls", type=float, default=0.1)
    parser.add_argument("--lambda-risk", type=float, default=0.0,
                        help="OCID-Grasp has no risk labels, keep 0 for OCID pretraining.")
    parser.add_argument("--freeze-base-epochs", type=int, default=2,
                        help="Train object semantic head first without disturbing grasp backbone.")

    parser.add_argument("--use-depth", type=int, default=1)
    parser.add_argument("--use-rgb", type=int, default=1)
    parser.add_argument("--use-dropout", type=int, default=0)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--channel-size", type=int, default=32)
    parser.add_argument("--pretrained-base", type=str, default="",
                        help="Optional MFENet checkpoint. Supports state_dict or full torch-saved model.")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=123)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def load_pretrained_base(base_model, ckpt_path):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading pretrained MFENet base: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "base_model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    missing, unexpected = base_model.load_state_dict(cleaned, strict=False)
    print("[INFO] Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
    print("[INFO] Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")


def set_base_trainable(model, trainable):
    for p in model.base_model.parameters():
        p.requires_grad = bool(trainable)


def make_optimizer(model, lr_base, lr_head, weight_decay):
    base_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("base_model."):
            base_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": lr_base})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})

    return optim.AdamW(groups, weight_decay=weight_decay)


def batch_to_device(batch, device):
    target = {
        "pos": batch["pos"].to(device),
        "cos": batch["cos"].to(device),
        "sin": batch["sin"].to(device),
        "width": batch["width"].to(device),
        "object_labels": batch["object_labels"].to(device),
    }
    return batch["input"].to(device), batch["masks"].to(device), target


def run_epoch(model, loader, criterion, optimizer, device, train=True, batches_per_epoch=0):
    model.train(train)
    total = {}
    count = 0

    for batch_idx, batch in enumerate(loader):
        if batches_per_epoch > 0 and batch_idx >= batches_per_epoch:
            break

        x, masks, target = batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            pred = model(x, masks=masks)
            loss, loss_dict = criterion(pred, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        for k, v in loss_dict.items():
            total[k] = total.get(k, 0.0) + float(v)
        count += 1

        if train and batch_idx % 20 == 0:
            print(
                f"[TRAIN] batch={batch_idx:04d} "
                f"loss={loss_dict['loss_total']:.4f} "
                f"grasp={loss_dict['loss_grasp']:.4f} "
                f"cls={loss_dict['loss_cls']:.4f}"
            )

    if count == 0:
        return {k: 0.0 for k in ["loss_total", "loss_grasp", "loss_cls", "loss_risk"]}
    return {k: v / count for k, v in total.items()}


def save_checkpoint(save_path, model, optimizer, epoch, args, metrics):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(payload, save_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = build_device(args.cpu)
    print("[INFO] device:", device)

    input_channels = int(args.use_depth) + 3 * int(args.use_rgb)
    if input_channels <= 0:
        raise ValueError("At least one of --use-depth/--use-rgb must be enabled.")

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    run_name = f"{timestamp}_{args.description}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tb = tensorboardX.SummaryWriter(str(save_dir)) if tensorboardX is not None else None

    train_set = OCIDGraspObjectDataset(
        root=args.dataset_root,
        split="train",
        output_size=args.input_size,
        max_objects=args.max_objects,
        include_depth=bool(args.use_depth),
        include_rgb=bool(args.use_rgb),
        max_grasps=args.max_grasps,
    )
    val_set = OCIDGraspObjectDataset(
        root=args.dataset_root,
        split="val",
        output_size=args.input_size,
        max_objects=args.max_objects,
        include_depth=bool(args.use_depth),
        include_rgb=bool(args.use_rgb),
        max_grasps=args.max_grasps,
    )

    print(f"[INFO] train size: {len(train_set)}")
    print(f"[INFO] val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    base_model = MFENet(
        input_channels=input_channels,
        dropout=bool(args.use_dropout),
        prob=args.dropout_prob,
        channel_size=args.channel_size,
    )
    load_pretrained_base(base_model, args.pretrained_base)

    model = ObjectSemanticMFENet(
        base_model=base_model,
        num_classes=args.num_classes,
        num_risk_levels=args.num_risk_levels,
        feature_key=args.feature_key,
        hidden_dim=128,
        enable_risk=False,  # OCID-Grasp has no risk labels. Enable risk during industrial fine-tuning.
    ).to(device)

    criterion = ObjectSemanticGraspLoss(
        lambda_cls=args.lambda_cls,
        lambda_risk=args.lambda_risk,
    ).to(device)

    best_val = float("inf")

    for epoch in range(args.epochs):
        freeze_base = epoch < args.freeze_base_epochs
        set_base_trainable(model, not freeze_base)
        optimizer = make_optimizer(model, args.lr_base, args.lr_head, args.weight_decay)

        print("=" * 80)
        print(f"[INFO] Epoch {epoch + 1}/{args.epochs} | freeze_base={freeze_base}")

        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, batches_per_epoch=args.batches_per_epoch,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, batches_per_epoch=0,
        )

        print("[INFO] train:", train_metrics)
        print("[INFO] val:", val_metrics)

        if tb is not None:
            for k, v in train_metrics.items():
                tb.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                tb.add_scalar(f"val/{k}", v, epoch)

        latest_path = save_dir / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, args, {"train": train_metrics, "val": val_metrics})

        val_loss = val_metrics.get("loss_total", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            best_path = save_dir / "best.pth"
            save_checkpoint(best_path, model, optimizer, epoch, args, {"train": train_metrics, "val": val_metrics})
            print(f"[INFO] saved best checkpoint: {best_path}")

    if tb is not None:
        tb.close()
    print(f"[DONE] Training finished. Logs saved to {save_dir}")


if __name__ == "__main__":
    main()
