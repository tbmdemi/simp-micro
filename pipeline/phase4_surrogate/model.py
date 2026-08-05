"""
Phase 4 - model.py
====================
CNN baseline: 4x Conv(3x3)+BN+ReLU+MaxPool -> GAP -> concat seed one-hot
-> 2 FC layer -> 3 output (v12, v21, volfrac).

Nếu R² < 0.90, thử tăng CHANNELS (VD [32,64,128,256] -> [64,128,256,512])
hoặc thêm residual connection - chỉ cần đổi tham số truyền vào
SurrogateCNN(), không cần đổi cấu trúc file.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class SurrogateCNN(nn.Module):
    def __init__(self, n_seeds: int, channels=(32, 64, 128, 256), fc_hidden=128,
                 n_outputs: int = 3):
        """n_outputs=3 (mac dinh, tuong thich nguoc): [v12,v21,volfrac_achieved].
        n_outputs=5: them f1=E11/E0, f2=E22/E0 (backfill 2026-08-05, xem
        dataset.py::AuxeticDataset(include_f1f2=True))."""
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, channels[-1], 1, 1)

        fc_in = channels[-1] + n_seeds  # concat seed one-hot sau GAP
        self.n_outputs = n_outputs
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, n_outputs),
        )

    def forward(self, image, seed_vec):
        x = self.conv(image)                # (B, C, H', W')
        x = self.gap(x).flatten(1)           # (B, C)
        x = torch.cat([x, seed_vec], dim=1)  # (B, C + n_seeds)
        return self.fc(x)                    # (B, n_outputs)


if __name__ == "__main__":
    # Self-test: kiểm tra forward pass chạy đúng shape trước khi viết train.py
    model = SurrogateCNN(n_seeds=11)
    dummy_img = torch.randn(4, 1, 64, 64)
    dummy_seed = torch.zeros(4, 11)
    dummy_seed[:, 0] = 1.0
    out = model(dummy_img, dummy_seed)
    print(f"Output shape: {out.shape}  (kỳ vọng: [4, 3])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n_params:,}")