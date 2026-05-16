import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectMaskPooling(nn.Module):
    """
    Object-centric mask feature pooling.

    Args:
        feat:  [B, C, Hf, Wf]
        masks: [B, N, Hm, Wm]

    Returns:
        object_feats:  [B, N, C]
        valid_objects: [B, N]
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat, masks):
        if feat.dim() != 4:
            raise ValueError(f"feat should be [B,C,H,W], got {feat.shape}")
        if masks.dim() != 4:
            raise ValueError(f"masks should be [B,N,H,W], got {masks.shape}")

        b, c, hf, wf = feat.shape
        bm, n, _, _ = masks.shape
        if b != bm:
            raise ValueError(f"Batch mismatch: feat B={b}, masks B={bm}")

        masks = masks.float()
        masks_resized = F.interpolate(masks, size=(hf, wf), mode="nearest")
        mask_area = masks_resized.sum(dim=(2, 3))
        valid_objects = mask_area > self.eps

        feat_expand = feat.unsqueeze(1)              # [B,1,C,Hf,Wf]
        mask_expand = masks_resized.unsqueeze(2)     # [B,N,1,Hf,Wf]

        pooled = (feat_expand * mask_expand).sum(dim=(3, 4))
        denom = mask_expand.sum(dim=(3, 4)).clamp_min(self.eps)
        object_feats = pooled / denom

        return object_feats, valid_objects
