import torch
import torch.nn as nn

from inference.models.object_mask_pooling import ObjectMaskPooling
from inference.models.object_semantic_head import ObjectSemanticHead
from inference.models.risk_head import ObjectRiskHead
from inference.models.mask_guided_grasp import mask_guided_grasp_select


class ObjectSemanticMFENet(nn.Module):
    """
    MFENet + object-centric semantic branch + risk head + mask-guided grasp selection.

    This wrapper preserves the original MFENet training pipeline because the base MFENet
    still returns only pos/cos/sin/width by default. The semantic/risk branches are used
    when masks are provided.

    Args:
        base_model: MFENet instance.
        feature_key: p2/p3/p4/c2/c3/c4/grasp_feat.
            p2 is recommended for object-level semantics because it is FPN-fused and
            still has relatively high spatial resolution.
    """

    def __init__(
        self,
        base_model,
        num_classes=31,
        num_risk_levels=3,
        feature_key="p2",
        hidden_dim=128,
        enable_risk=True,
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_key = feature_key
        self.enable_risk = enable_risk

        self.mask_pooling = ObjectMaskPooling()
        self.semantic_head = ObjectSemanticHead(num_classes=num_classes, hidden_dim=hidden_dim)
        self.risk_head = ObjectRiskHead(num_risk_levels=num_risk_levels, hidden_dim=hidden_dim)

    def _select_feature(self, aux_features):
        if self.feature_key in ["p2", "p3", "p4"]:
            return aux_features["p_features"][self.feature_key]
        if self.feature_key in ["c2", "c3", "c4"]:
            return aux_features["features"][self.feature_key]
        if self.feature_key == "grasp_feat":
            return aux_features["grasp_feat"]
        raise ValueError(
            f"Unknown feature_key={self.feature_key}. "
            f"Choose from p2,p3,p4,c2,c3,c4,grasp_feat."
        )

    def forward(self, x, masks=None):
        outputs = self.base_model(x, return_features=True)
        if len(outputs) != 5:
            raise RuntimeError(
                "base_model must return pos, cos, sin, width, aux_features "
                "when return_features=True."
            )

        pos, cos, sin, width, aux_features = outputs
        out = {
            "pos": pos,
            "cos": cos,
            "sin": sin,
            "width": width,
        }

        if masks is not None:
            feat = self._select_feature(aux_features)
            object_feats, valid_objects = self.mask_pooling(feat, masks)
            object_logits = self.semantic_head(object_feats)

            grasp_result = mask_guided_grasp_select(
                pos=pos,
                cos=cos,
                sin=sin,
                width=width,
                masks=masks,
            )

            out.update({
                "object_logits": object_logits,
                "valid_objects": valid_objects,
                "grasp_result": grasp_result,
            })

            if self.enable_risk:
                out["risk_logits"] = self.risk_head(object_feats)

        return out
