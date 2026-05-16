import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectSemanticGraspLoss(nn.Module):
    """
    Multi-task loss for object-centric semantic MFENet.

    L_total = L_grasp + lambda_cls * L_cls + lambda_risk * L_risk

    For OCID-Grasp, pass object_labels and omit risk_labels.
    For industrial fine-tuning, pass both object_labels and risk_labels.
    """

    def __init__(self, lambda_cls=0.1, lambda_risk=0.05):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_risk = lambda_risk
        self.smooth_l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()

    def _resize_like(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return pred

    def _object_ce_loss(self, logits, labels, valid_objects=None):
        b, n, c = logits.shape
        logits = logits.reshape(b * n, c)
        labels = labels.reshape(b * n).long()

        valid = labels >= 0
        if valid_objects is not None:
            valid = valid & valid_objects.reshape(b * n).bool()

        if valid.sum() == 0:
            return logits.sum() * 0.0
        return self.ce(logits[valid], labels[valid])

    def forward(self, pred, target):
        pos_p = self._resize_like(pred["pos"], target["pos"])
        cos_p = self._resize_like(pred["cos"], target["cos"])
        sin_p = self._resize_like(pred["sin"], target["sin"])
        width_p = self._resize_like(pred["width"], target["width"])

        loss_pos = self.smooth_l1(pos_p, target["pos"])
        loss_cos = self.smooth_l1(cos_p, target["cos"])
        loss_sin = self.smooth_l1(sin_p, target["sin"])
        loss_width = self.smooth_l1(width_p, target["width"])
        loss_grasp = loss_pos + loss_cos + loss_sin + loss_width

        loss_cls = pred["pos"].sum() * 0.0
        loss_risk = pred["pos"].sum() * 0.0

        if "object_logits" in pred and "object_labels" in target:
            loss_cls = self._object_ce_loss(
                pred["object_logits"],
                target["object_labels"],
                pred.get("valid_objects", None),
            )

        if "risk_logits" in pred and "risk_labels" in target:
            loss_risk = self._object_ce_loss(
                pred["risk_logits"],
                target["risk_labels"],
                pred.get("valid_objects", None),
            )

        loss_total = loss_grasp + self.lambda_cls * loss_cls + self.lambda_risk * loss_risk

        loss_dict = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_grasp": float(loss_grasp.detach().cpu()),
            "loss_cls": float(loss_cls.detach().cpu()),
            "loss_risk": float(loss_risk.detach().cpu()),
            "loss_pos": float(loss_pos.detach().cpu()),
            "loss_cos": float(loss_cos.detach().cpu()),
            "loss_sin": float(loss_sin.detach().cpu()),
            "loss_width": float(loss_width.detach().cpu()),
        }
        return loss_total, loss_dict
