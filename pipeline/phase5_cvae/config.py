"""
Phase 5 - config.py
=====================
Gom toàn bộ tham số cVAE về 1 chỗ (theo đúng pattern pipeline/params.py đã
dùng ở Phase 1/2), thay vì rải rác giữa CLI args (train.py), hằng số hardcode
trong losses.py (PROP_LOSS_SCALE) và model.py (channels).

Cách dùng:
    từ CLI (không đổi cách chạy cũ):
        python3 pipeline/phase5_cvae/train.py --gamma 20

    từ code / notebook, load 1 config đã lưu để tái lập chính xác:
        from config import CVAEConfig
        cfg = CVAEConfig.load("outputs/phase5/config_gamma20.json")

    lưu lại config đã dùng để train (train.py tự làm việc này, xem cuối file):
        cfg.save("outputs/phase5/config_gamma20.json")
"""
from dataclasses import dataclass, asdict, field
import json
from typing import Tuple


@dataclass
class CVAEConfig:
    # --- kiến trúc model ---
    latent_dim: int = 32
    condition_dim: int = 2
    resolution: int = 64
    channels: Tuple[int, ...] = (32, 64, 128, 256)   # trước đây hardcode trong model.py

    # --- loss ---
    kl_warmup: int = 30
    gamma: float = 20.0          # ĐÃ ĐỔI default 1.0 -> 20.0, theo kết quả sweep thật
                                  # (gamma=1: R2=-0.42, gamma=5: R2=0.45, gamma=20: R2=0.63)
    prop_loss_scale: float = 1000.0   # trước đây hardcode trong losses.py

    # --- training ---
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-5
    patience: int = 15

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CVAEConfig":
        with open(path) as f:
            data = json.load(f)
        data["channels"] = tuple(data["channels"])
        return cls(**data)


if __name__ == "__main__":
    # Self-test
    cfg = CVAEConfig()
    print(cfg)