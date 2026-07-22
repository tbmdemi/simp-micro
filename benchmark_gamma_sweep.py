"""
Benchmark thời gian train Phase 5 cVAE trên GPU thật (RTX 3050) trước khi
chạy full gamma sweep (10/30/50, 50 epochs).

Cách chạy (từ thư mục gốc repo, sau khi đã có outputs/phase3/*.npz và
outputs/phase4/surrogate_for_phase5.pt):

    python3 benchmark_gamma_sweep.py

Kết quả in ra: thời gian trung bình / epoch (train + val), và ước tính
tổng thời gian cho sweep gamma=10/30/50 x 50 epochs.
"""
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from model import CVAE                                   # noqa: E402
from dataset import CVAEDataset                           # noqa: E402
from losses import cvae_loss, kl_beta_schedule, load_frozen_surrogate  # noqa: E402
from config import CVAEConfig                              # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = CVAEConfig()  # batch_size=64, epochs=100 mặc định

    train_ds = CVAEDataset(os.path.join(PHASE3_DIR, "train.npz"))
    val_ds = CVAEDataset(os.path.join(PHASE3_DIR, "val.npz"))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=2)
    print(f"Train: {len(train_ds)} mau ({len(train_loader)} batch) | "
          f"Val: {len(val_ds)} mau ({len(val_loader)} batch)")

    model = CVAE(condition_dim=2, latent_dim=cfg.latent_dim,
                 resolution=cfg.resolution, channels=cfg.channels).to(device)
    surrogate, target_names = load_frozen_surrogate(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def run_epoch(loader, gamma, train):
        model.train(mode=train)
        for image, condition, seed_vec, _volfrac in loader:
            image, condition, seed_vec = (
                image.to(device), condition.to(device), seed_vec.to(device)
            )
            with torch.set_grad_enabled(train):
                recon, mu, logvar = model(image, condition, deterministic=not train)
                losses = cvae_loss(recon, image, mu, logvar, condition, seed_vec,
                                    surrogate, target_names, beta=1.0, gamma=gamma,
                                    prop_loss_scale=cfg.prop_loss_scale)
                if train:
                    optimizer.zero_grad()
                    losses["total"].backward()
                    optimizer.step()

    # warmup (loại trừ overhead CUDA init / cuDNN autotune lần đầu)
    run_epoch(train_loader, gamma=10.0, train=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    N_TIMED_EPOCHS = 3
    t0 = time.perf_counter()
    for _ in range(N_TIMED_EPOCHS):
        run_epoch(train_loader, gamma=10.0, train=True)
        run_epoch(val_loader, gamma=10.0, train=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sec_per_epoch = elapsed / N_TIMED_EPOCHS
    print(f"\n=> Trung binh {sec_per_epoch:.2f} giay / epoch (train+val)")

    epochs_per_run = 50
    n_gammas = 3
    total_sec = sec_per_epoch * epochs_per_run * n_gammas
    print(f"Uoc tinh 1 run (50 epoch): {sec_per_epoch*epochs_per_run/60:.1f} phut")
    print(f"Uoc tinh CA SWEEP gamma=10/30/50 (50 epoch x 3): "
          f"{total_sec/60:.1f} phut (~{total_sec/3600:.2f} gio)")
    print("Luu y: gia dinh khong co early stopping kich hoat som hon patience=15.")


if __name__ == "__main__":
    main()
