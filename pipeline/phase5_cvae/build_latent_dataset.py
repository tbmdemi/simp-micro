"""
Phase 5 - build_latent_dataset.py
============================================================
Giai đoạn 2 (plan.md, bước 1 "Latent diffusion prior"): mã hoá TOÀN BỘ
dataset (train/val/test) bằng ENCODER đã đóng băng của 1 checkpoint cVAE cho
sẵn, thu `z=mu` (deterministic, KHÔNG sample qua reparameterization) cho mỗi
mẫu. Đây là dữ liệu latent thật (aggregate posterior) sẽ dùng để train 1
diffusion model nhỏ thay cho `torch.randn(latent_dim)` mặc định của
`CVAE.generate()` - KHÔNG train gì ở đây, chỉ 1 lần forward-pass qua encoder.

Vì sao dùng `mu` (không sample `z~N(mu,var)` như lúc train VAE): `mu` là ước
lượng điểm tốt nhất của vị trí latent ứng với ảnh thật - cộng thêm nhiễu
`logvar` sẽ làm nhoè phân bố mà diffusion model phải học, trong khi mục tiêu
ở đây là học đúng HÌNH DẠNG manifold thật của `mu` (aggregate posterior),
không phải nhiễu tái tạo riêng của từng lần sample VAE. `logvar` vẫn được
lưu lại (không dùng ngay) để tham khảo độ "chắc chắn" của encoder từng mẫu.

Cách chạy:
    python3 pipeline/phase5_cvae/build_latent_dataset.py \\
        --cvae-ckpt outputs/phase5/cvae_kan_realphysics_v2.pt \\
        --data-dir outputs/phase3 --split train
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from adversarial_dataset import load_cvae  # noqa: E402
from dataset import CVAEDataset, condition_flags_from_dim  # noqa: E402

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


@torch.no_grad()
def encode_dataset(
    cvae_ckpt_path: str,
    npz_path: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """Mã hoá toàn bộ `npz_path` bằng encoder đóng băng của `cvae_ckpt_path`.

    Args:
        cvae_ckpt_path: checkpoint cVAE đã train (chỉ dùng phần encoder,
            đóng băng - không cập nhật gradient).
        npz_path: file `.npz` (train/val/test) đúng format `CVAEDataset`.
        device: "cpu" hoặc "cuda".
        batch_size: batch size khi forward qua encoder (không ảnh hưởng kết
            quả, chỉ ảnh hưởng tốc độ/bộ nhớ).

    Returns:
        dict {"mu": (N,latent_dim), "logvar": (N,latent_dim),
        "condition": (N,condition_dim), "v12": (N,), "v21": (N,)} - toàn bộ
        `numpy.ndarray`, sẵn sàng lưu bằng `np.savez`.
    """
    ckpt_meta = torch.load(
        cvae_ckpt_path, map_location="cpu", weights_only=False
    )
    condition_dim = ckpt_meta.get("condition_dim", 2)
    del ckpt_meta
    extended_condition, include_nu0 = condition_flags_from_dim(condition_dim)

    model = load_cvae(cvae_ckpt_path, device)
    model.eval()

    ds = CVAEDataset(
        npz_path,
        extended_condition=extended_condition,
        include_nu0=include_nu0,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    mus, logvars, conditions = [], [], []
    for image, condition, _seed_vec, _volfrac in loader:
        image = image.to(device)
        condition = condition.to(device)
        mu, logvar = model.encoder(image, condition)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        conditions.append(condition.cpu().numpy())

    return {
        "mu": np.concatenate(mus, axis=0),
        "logvar": np.concatenate(logvars, axis=0),
        "condition": np.concatenate(conditions, axis=0),
        "v12": ds.v12,
        "v21": ds.v21,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cvae-ckpt",
        default=os.path.join(PHASE5_DIR, "cvae_kan_realphysics_v2.pt"),
    )
    parser.add_argument("--data-dir", default=PHASE3_DIR)
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Mặc định 'cpu' để không tranh chấp GPU với job train đang "
        "chạy nền - việc này chỉ forward-pass 1 lần, không cần GPU.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    npz_path = os.path.join(args.data_dir, f"{args.split}.npz")
    result = encode_dataset(
        args.cvae_ckpt, npz_path, args.device, args.batch_size
    )

    ckpt_name = os.path.splitext(os.path.basename(args.cvae_ckpt))[0]
    out_path = args.out or os.path.join(
        PHASE5_DIR, f"{ckpt_name}_latents_{args.split}.npz"
    )
    np.savez(out_path, **result)
    print(
        f"Đã lưu {result['mu'].shape[0]} latent vector "
        f"(dim={result['mu'].shape[1]}) -> {out_path}"
    )
    print(
        f"mu: mean={result['mu'].mean():.4f}, std={result['mu'].std():.4f} "
        f"(so với N(0,1) prior mặc định)"
    )


if __name__ == "__main__":
    main()
