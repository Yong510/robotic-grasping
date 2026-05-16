import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.models.mfenet import MFENet
from inference.models.object_semantic_mfenet import ObjectSemanticMFENet
from losses.object_semantic_grasp_loss import ObjectSemanticGraspLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    batch_size = 2
    input_channels = 4
    image_size = 300
    num_objects = 5
    num_classes = 31
    num_risk_levels = 3

    base_model = MFENet(
        input_channels=input_channels,
        dropout=False,
        prob=0.1,
        channel_size=32,
    )

    model = ObjectSemanticMFENet(
        base_model=base_model,
        num_classes=num_classes,
        num_risk_levels=num_risk_levels,
        feature_key="p2",
        hidden_dim=128,
        enable_risk=True,
    ).to(device)

    x = torch.randn(batch_size, input_channels, image_size, image_size).to(device)

    masks = torch.zeros(batch_size, num_objects, image_size, image_size).to(device)
    for b in range(batch_size):
        for n in range(num_objects):
            y1 = 20 + n * 30
            y2 = min(y1 + 50, image_size)
            x1 = 30 + n * 25
            x2 = min(x1 + 60, image_size)
            masks[b, n, y1:y2, x1:x2] = 1.0

    pred = model(x, masks=masks)

    print("[OK] forward success")
    print("[INFO] pos:", tuple(pred["pos"].shape))
    print("[INFO] cos:", tuple(pred["cos"].shape))
    print("[INFO] sin:", tuple(pred["sin"].shape))
    print("[INFO] width:", tuple(pred["width"].shape))
    print("[INFO] object_logits:", tuple(pred["object_logits"].shape))
    print("[INFO] risk_logits:", tuple(pred["risk_logits"].shape))
    print("[INFO] valid_objects:", tuple(pred["valid_objects"].shape))
    print("[INFO] grasp scores:", tuple(pred["grasp_result"]["scores"].shape))

    assert pred["object_logits"].shape == (batch_size, num_objects, num_classes)
    assert pred["risk_logits"].shape == (batch_size, num_objects, num_risk_levels)

    target = {
        "pos": torch.randn_like(pred["pos"]),
        "cos": torch.randn_like(pred["cos"]),
        "sin": torch.randn_like(pred["sin"]),
        "width": torch.randn_like(pred["width"]),
        "object_labels": torch.randint(0, num_classes, (batch_size, num_objects), device=device),
        "risk_labels": torch.randint(0, num_risk_levels, (batch_size, num_objects), device=device),
    }

    criterion = ObjectSemanticGraspLoss(lambda_cls=0.1, lambda_risk=0.05).to(device)
    loss, loss_dict = criterion(pred, target)
    print("[INFO] loss:", loss_dict)

    loss.backward()
    print("[OK] backward success")
    print("[OK] all checks passed")


if __name__ == "__main__":
    main()
