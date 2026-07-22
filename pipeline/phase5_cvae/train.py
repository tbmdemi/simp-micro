"""
Phase 5 - train.py  (roadmap bước 5.4)
=========================================
Train cVAE baseline trên outputs/phase3/train.npz (đã augment, 33,120 mẫu),
validate trên val.npz, dùng surrogate frozen (Phase 4) cho property loss.

Cách chạy:
    python3 pipeline/phase5_cvae/train.py

Tham số hay chỉnh:
    --latent-dim 32        kích thước latent z
    --epochs 100
    --batch-size 64
    --kl-warmup 30          số epoch để beta KL tăng 0 -> 1 (annealing)
    --gamma 1.0              trọng số property-consistency loss
    --lr 1e-3

Output:
    outputs/phase5/cvae_best.pt        checkpoint tốt nhất theo val loss
    outputs/phase5/train_history.json  lịch sử loss từng epoch (để vẽ đồ thị)

YÊU CẦU TRƯỚC KHI CHẠY:
    outputs/phase4/surrogate_for_phase5.pt phải tồn tại - nếu chưa có, chạy
    trước: python3 pipeline/phase4_surrogate/export_for_phase5.py
"""
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import CVAE                     # noqa: E402
from dataset import CVAEDataset            # noqa: E402
from losses import (                       # noqa: E402
    cvae_loss, kl_beta_schedule, load_frozen_surrogate,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


def run_epoch(model, loader, surrogate, target_names, optimizer, beta, gamma,
              lambda_tv, lambda_bin, device, train: bool):
    model.train(mode=train)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "prop": 0.0,
              "prop_weighted": 0.0, "tv": 0.0, "binarization": 0.0}
    n = 0
    for image, condition, seed_vec, _volfrac in loader:
        image = image.to(device)
        condition = condition.to(device)
        seed_vec = seed_vec.to(device)
        bsz = image.size(0)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(image, condition, deterministic=not train)
            losses = cvae_loss(
                recon, image, mu, logvar, condition, seed_vec,
                surrogate, target_names, beta=beta, gamma=gamma,
                lambda_tv=lambda_tv, lambda_bin=lambda_bin,
            )
            if train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

        totals["total"] += losses["total"].item() * bsz
        totals["recon"] += losses["recon"].item() * bsz
        totals["kl"] += losses["kl"].item() * bsz
        totals["prop"] += losses["prop"].item() * bsz
        totals["prop_weighted"] += losses["prop_weighted"].item() * bsz
        totals["tv"] += losses["tv"].item() * bsz
        totals["binarization"] += losses["binarization"].item() * bsz
        n += bsz

    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kl-warmup", type=int, default=30,
                         help="số epoch để beta KL tăng tuyến tính 0 -> 1")
    parser.add_argument("--gamma", type=float, default=1.0,
                         help="trọng số property-consistency loss")
    parser.add_argument("--lambda-tv", type=float, default=0.0,
                         help="trọng số total-variation regularization (chống nhiễu/checkerboard). "
                              "Mặc định 0.0 (tắt) để giữ tương thích baseline cũ.")
    parser.add_argument("--lambda-bin", type=float, default=0.0,
                         help="trọng số binarization loss (ép ảnh về gần nhị phân 0/1). "
                              "Mặc định 0.0 (tắt) để giữ tương thích baseline cũ.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5,
                         help="lr tối thiểu ở cuối CosineAnnealing (0 = tắt schedule, giữ lr cố định)")
    parser.add_argument("--patience", type=int, default=15,
                         help="early stopping: dừng nếu val loss không giảm sau N epoch")
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(PHASE5_DIR, exist_ok=True)

    train_ds = CVAEDataset(os.path.join(PHASE3_DIR, "train.npz"))
    val_ds = CVAEDataset(os.path.join(PHASE3_DIR, "val.npz"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)
    print(f"Train: {len(train_ds)} mẫu | Val: {len(val_ds)} mẫu")

    model = CVAE(condition_dim=2, latent_dim=args.latent_dim,
                 resolution=args.resolution).to(device)
    surrogate, target_names = load_frozen_surrogate(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_min > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )

    history = []
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        beta = kl_beta_schedule(epoch, args.kl_warmup, beta_max=1.0)

        train_stats = run_epoch(model, train_loader, surrogate, target_names,
                                 optimizer, beta, args.gamma,
                                 args.lambda_tv, args.lambda_bin, device, train=True)
        val_stats = run_epoch(model, val_loader, surrogate, target_names,
                               optimizer, beta, args.gamma,
                               args.lambda_tv, args.lambda_bin, device, train=False)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(f"[{epoch:03d}/{args.epochs}] beta={beta:.3f} lr={current_lr:.2e} | "
              f"train total={train_stats['total']:.2f} recon={train_stats['recon']:.2f} "
              f"kl={train_stats['kl']:.3f} prop={train_stats['prop']:.4f} "
              f"prop_w={train_stats['prop_weighted']:.2f} tv={train_stats['tv']:.4f} "
              f"bin={train_stats['binarization']:.4f} || "
              f"val total={val_stats['total']:.2f} recon={val_stats['recon']:.2f} "
              f"kl={val_stats['kl']:.3f} prop={val_stats['prop']:.4f} "
              f"prop_w={val_stats['prop_weighted']:.2f} tv={val_stats['tv']:.4f} "
              f"bin={val_stats['binarization']:.4f}")

        history.append({"epoch": epoch, "beta": beta,
                         "train": train_stats, "val": val_stats})

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "latent_dim": args.latent_dim,
                "condition_dim": 2,
                "resolution": args.resolution,
                "epoch": epoch,
                "val_loss": best_val,
                "gamma": args.gamma,
                "lambda_tv": args.lambda_tv,
                "lambda_bin": args.lambda_bin,
            }, os.path.join(PHASE5_DIR, "cvae_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping tại epoch {epoch} "
                      f"(val loss không giảm trong {args.patience} epoch)")
                break

    with open(os.path.join(PHASE5_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Đã lưu checkpoint tốt nhất: outputs/phase5/cvae_best.pt "
          f"(val_loss={best_val:.2f})")


if __name__ == "__main__":
    main()