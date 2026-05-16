import torch
import torch.nn as nn


class ObjectSemanticHead(nn.Module):
    """
    Object-level semantic classification head.

    Args:
        object_feats: [B, N, C]

    Returns:
        logits: [B, N, num_classes]
    """

    def __init__(self, num_classes=31, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, object_feats):
        if object_feats.dim() != 3:
            raise ValueError(f"object_feats should be [B,N,C], got {object_feats.shape}")
        return self.classifier(object_feats)
