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

try:
    from efficient_kan import EfficientKANLinear
except ModuleNotFoundError:
    # Direct script execution puts this directory on sys.path, not the repo
    # root where the vendored package lives.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from efficient_kan import EfficientKANLinear


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
                 n_outputs: int = 3, include_nu0: bool = False,
                 use_kan: bool = False):
        """n_outputs=3 (mac dinh, tuong thich nguoc): [v12,v21,volfrac_achieved].
        n_outputs=5: them f1=E11/E0, f2=E22/E0 (backfill 2026-08-05, xem
        dataset.py::AuxeticDataset(include_f1f2=True)).

        include_nu0: them nu (he so Poisson vat lieu nen) lam INPUT PHU, concat
        cung seed one-hot sau GAP - can thiet vi khi nu0 bien thien, quan he
        hinh hoc->tinh chat khong con la ham 1-1 cua anh mat do (Giai doan A,
        A4, docs/archive/PROJECT_PLAN.md Nhom 1). Mac dinh False - kien
        truc/forward()
        y het truoc day, tuong thich nguoc hoan toan voi checkpoint cu.

        use_kan: thay hai lop FC cua head bang EfficientKANLinear. Mac dinh
        False de checkpoint CNN cu tiep tuc load duoc; GAP duoc giu nguyen de
        khong lam tang kich thuoc dau vao cua KAN mot cach khong can thiet."""
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, channels[-1], 1, 1)

        self.include_nu0 = include_nu0
        self.use_kan = use_kan
        fc_in = channels[-1] + n_seeds + (1 if include_nu0 else 0)  # concat seed one-hot (+nu0) sau GAP
        self.n_outputs = n_outputs
        fc_layer = EfficientKANLinear if use_kan else nn.Linear
        self.fc = nn.Sequential(
            fc_layer(fc_in, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            fc_layer(fc_hidden, n_outputs),
        )

    def forward(self, image, seed_vec, nu0=None):
        x = self.conv(image)                # (B, C, H', W')
        x = self.gap(x).flatten(1)           # (B, C)
        parts = [x, seed_vec]
        if self.include_nu0:
            if nu0 is None:
                raise ValueError("include_nu0=True nhung forward() khong nhan nu0")
            parts.append(nu0.view(-1, 1).to(x.dtype))
        x = torch.cat(parts, dim=1)          # (B, C + n_seeds [+ 1])
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