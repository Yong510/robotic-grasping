import torch
import torch.nn.functional as F


def mask_guided_grasp_select(pos, cos, sin, width, masks):
    """
    Select the best grasp point inside each object mask.

    Args:
        pos/cos/sin/width: [B, 1, H, W]
        masks:             [B, N, Hm, Wm]

    Returns:
        dict with scores/xs/ys/angles/widths, each [B, N].
    """

    if pos.dim() != 4:
        raise ValueError(f"pos should be [B,1,H,W], got {pos.shape}")
    if masks.dim() != 4:
        raise ValueError(f"masks should be [B,N,H,W], got {masks.shape}")

    b, _, h, w = pos.shape
    bm, n, _, _ = masks.shape
    if b != bm:
        raise ValueError(f"Batch mismatch: pos B={b}, masks B={bm}")

    masks = masks.float()
    masks_resized = F.interpolate(masks, size=(h, w), mode="nearest")

    q = pos[:, 0]
    q_expand = q.unsqueeze(1).expand(b, n, h, w)
    masked_q = q_expand.masked_fill(masks_resized <= 0.5, -1e6)

    flat = masked_q.reshape(b, n, -1)
    scores, indices = flat.max(dim=-1)

    ys = indices // w
    xs = indices % w

    cos_map = cos[:, 0]
    sin_map = sin[:, 0]
    width_map = width[:, 0]

    batch_ids = torch.arange(b, device=pos.device).view(b, 1).expand(b, n)
    selected_cos = cos_map[batch_ids, ys, xs]
    selected_sin = sin_map[batch_ids, ys, xs]
    selected_width = width_map[batch_ids, ys, xs]

    angles = 0.5 * torch.atan2(selected_sin, selected_cos)

    return {
        "scores": scores,
        "xs": xs,
        "ys": ys,
        "angles": angles,
        "widths": selected_width,
    }
