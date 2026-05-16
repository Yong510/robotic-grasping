import torch
import sys
sys.path.insert(0, '.')
from inference.models.mfenet import MFENet

# 自动选择可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MFENet(input_channels=4, dropout=True, prob=0.1, channel_size=32).to(device)
dummy = torch.randn(2, 4, 300, 300).to(device)
pos, cos, sin, width = model(dummy)
print("Pos shape:", pos.shape)
print("Cos shape:", cos.shape)
print("Sin shape:", sin.shape)
print("Width shape:", width.shape)
loss = pos.mean() + cos.mean() + sin.mean() + width.mean()
loss.backward()
print("Backward pass successful!")
