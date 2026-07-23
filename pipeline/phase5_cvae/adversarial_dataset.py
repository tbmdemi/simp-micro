"""
Phase 5 - adversarial_dataset.py
============================================================
Sinh "mẫu đối kháng" cho self-play: lấy checkpoint cVAE hiện tại, generate
ảnh ở các condition trong phân phối train, chấm điểm THẬT bằng FE + hóa đồng
nhất (không qua surrogate), rồi đóng gói thành .npz cùng định dạng
outputs/phase3/*.npz để AuxeticDataset (Phase 4) load thẳng được, không cần
sửa gì ở dataset.py.

Tái dùng resize_to_fe_grid()/evaluate_density_field()/FE_PARAMS từ
verify_fe.py (đã sanity-check, KHÔNG viết lại logic FE ở đây) - xem cảnh báo
grid size trong verify_fe.py trước khi đổi FE_PARAMS.

Ảnh lưu vào "images" là bản LIÊN TỤC (chưa binarize) - đúng với những gì
surrogate thực sự thấy trong property_consistency_loss lúc train cVAE
(surrogate(recon, seed_vec) dùng recon liên tục, không binarize). Binarize
chỉ dùng nội bộ để tính v12/v21/volfrac THẬT làm nhãn train cho surrogate.

Cách chạy độc lập (debug):
    python3 pipeline/phase5_cvae/adversarial_dataset.py \\
        --cvae-ckpt outputs/phase5/gamma_sweep_results/cvae_best_gamma20.pt \\
        --out outputs/phase5/self_play/round1/adversarial.npz \\
        --n-conditions 8 --seeds-per-condition 2
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import CVAE                                        # noqa: E402
from verify_fe import (                                        # noqa: E402
    FE_PARAMS, resize_to_fe_grid, evaluate_density_field,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


def load_cvae(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE(
        condition_dim=ckpt.get("condition_dim", 2),
        latent_dim=ckpt["latent_dim"],
        resolution=ckpt.get("resolution", 64),
        channels=ckpt.get("channels", (32, 64, 128, 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate_adversarial_npz(
    cvae_ckpt_path: str,
    out_path: str,
    n_conditions: int = 8,
    seeds_per_condition: int = 2,
    device: str = "cpu",
    penal: float = 3.0,
    seed: int = 0,
):
    """Sinh (n_conditions * seeds_per_condition) mẫu đối kháng, lưu .npz
    cùng schema outputs/phase3/*.npz. seed_classes lấy đúng thứ tự từ
    train.npz để cột one-hot khớp với AuxeticDataset gốc."""
    train_raw = np.load(os.path.join(PHASE3_DIR, "train.npz"), allow_pickle=True)
    seed_classes = train_raw["seed_classes"]
    n_seeds = len(seed_classes)
    n_seeds_use = min(seeds_per_condition, n_seeds)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(train_raw["v12"]), size=n_conditions, replace=False)
    conditions = np.stack([train_raw["v12"][idxs], train_raw["v21"][idxs]], axis=1)
    seed_order = rng.permutation(n_seeds)[:n_seeds_use]

    model = load_cvae(cvae_ckpt_path, device)

    images, v12s, v21s, volfracs, onehots = [], [], [], [], []
    fe_params = dict(FE_PARAMS, penal=penal)

    print(f"Sinh {n_conditions} conditions x {n_seeds_use} seeds = "
          f"{n_conditions * n_seeds_use} mẫu đối kháng từ {cvae_ckpt_path}...")

    for cond in conditions:
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        for seed_idx in seed_order:
            with torch.no_grad():
                img = model.generate(cond_t, n_samples=1, device=device)
            img64 = img.squeeze().cpu().numpy().astype(np.float32)  # liên tục [0,1]

            img_bin = (img64 > 0.5).astype(np.float32)
            img_fe = resize_to_fe_grid(img_bin, fe_params["nely"], fe_params["nelx"])
            try:
                v12_real, v21_real, _ = evaluate_density_field(img_fe, fe_params)
            except Exception as e:
                print(f"  [bỏ qua] condition={cond} seed_idx={seed_idx}: {e}")
                continue
            volfrac_real = float(img_fe.mean())

            onehot = np.zeros(n_seeds, dtype=np.float32)
            onehot[seed_idx] = 1.0

            images.append(img64)
            v12s.append(v12_real)
            v21s.append(v21_real)
            volfracs.append(volfrac_real)
            onehots.append(onehot)

    if not images:
        raise RuntimeError("Không sinh được mẫu đối kháng nào (mọi FE solve đều lỗi).")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        images=np.stack(images),
        v12=np.array(v12s, dtype=np.float32),
        v21=np.array(v21s, dtype=np.float32),
        volfrac_achieved=np.array(volfracs, dtype=np.float32),
        seed_onehot=np.stack(onehots),
        seed_classes=seed_classes,
    )
    print(f"Đã lưu {len(images)} mẫu đối kháng -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-conditions", type=int, default=8)
    parser.add_argument("--seeds-per-condition", type=int, default=2)
    parser.add_argument("--penal", type=float, default=3.0,
                         help="penal đại diện (xấp xỉ - ảnh sinh ra không có "
                              "penal gốc, xem docstring verify_fe.py)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_adversarial_npz(
        args.cvae_ckpt, args.out, args.n_conditions, args.seeds_per_condition,
        device, args.penal, args.seed,
    )


if __name__ == "__main__":
    main()
