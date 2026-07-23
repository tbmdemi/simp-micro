"""
Phase 4 - train.py
========================================
Huấn luyện SurrogateCNN trên train.npz, theo dõi val loss, early stopping.

Cách chạy:
    python3 pipeline/phase4_surrogate/train.py
    python3 pipeline/phase4_surrogate/train.py --epochs 100 --batch_size 256

Muốn đổi gì:
  - Đổi trọng số loss giữa v12/v21/volfrac -> sửa LOSS_WEIGHTS bên dưới
  - Đổi optimizer/lr -> sửa trong hàm main(), phần "optimizer = ..."
  - Đổi kiến trúc (Phương án B) -> sửa tham số channels khi khởi tạo
    SurrogateCNN trong hàm main()
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AuxeticDataset
from model import SurrogateCNN

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE4_DIR = os.path.join(REPO_ROOT, "outputs", "phase4")

# Trọng số loss theo roadmap 4.1: v12, v21 quan trọng hơn volfrac
LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 0.3])


def weighted_mse(pred, target, weights):
    """Compute weighted MSE.

    L1: Clamp to [-10, 10] std to prevent NaN from extreme values (early training
    or degenerate inputs).
    """
    # ── L1: prevent NaN from extreme deviations ──
    pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-10.0)
    diff = pred - target
    diff = diff.clamp(-10.0, 10.0)  # prevent squared blowup
    per_target_mse = (diff ** 2).mean(dim=0)  # (3,)
    return (per_target_mse * weights.to(pred.device)).sum(), per_target_mse.detach()


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_per_target = torch.zeros(3)
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for image, seed_vec, targets in loader:
            image, seed_vec, targets = image.to(device), seed_vec.to(device), targets.to(device)
            pred = model(image, seed_vec)
            loss, per_target = weighted_mse(pred, targets, LOSS_WEIGHTS)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Giới hạn norm gradient <= 1.0 - không có clip này val_loss
                # dao động mạnh giữa các epoch dù train_loss giảm đều.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_per_target += per_target.cpu()
            n_batches += 1

    return total_loss / n_batches, total_per_target / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4,
                         help="LR ban đầu (1e-3 gây val_loss dao động mạnh giữa epoch).")
    parser.add_argument("--patience", type=int, default=10,
                         help="Số epoch chờ trước khi early-stop nếu val loss không cải thiện")
    parser.add_argument("--limit", type=int, default=None,
                         help="Chỉ dùng N mẫu đầu (debug nhanh, bỏ trống = toàn bộ)")
    parser.add_argument("--adversarial-npz", type=str, nargs="*", default=None,
                         help="Đường dẫn .npz bổ sung (cùng schema outputs/phase3), "
                              "vd mẫu đối kháng từ adversarial_dataset.py, nối thêm "
                              "vào train set qua ConcatDataset (self-play).")
    parser.add_argument("--adversarial-oversample", type=int, default=1,
                         help="Lặp lại mỗi dataset --adversarial-npz N lần trước khi "
                              "nối vào train set. Vài chục mẫu đối kháng giữa 33k mẫu "
                              "thật chỉ chiếm ~0.1%% mỗi batch - gradient signal gần "
                              "như vô hình, surrogate không thực sự học được gì mới "
                              "(xác nhận bằng cách re-verify checkpoint round0-8 trên "
                              "cùng 1 tập test cố định: R2 không cải thiện thật, chỉ "
                              "dao động do nhiễu). Tăng số này (vd 30-50) để mẫu đối "
                              "kháng chiếm tỉ trọng đáng kể trong gradient mỗi epoch.")
    parser.add_argument("--init-from", type=str, default=None,
                         help="Checkpoint .pt để load model_state_dict trước khi train "
                              "(fine-tune tiếp thay vì train từ đầu, dùng cho self-play).")
    parser.add_argument("--output-name", type=str, default="surrogate_best.pt",
                         help="Tên file checkpoint lưu trong outputs/phase4/ - đổi "
                              "tên này để không ghi đè surrogate_best.pt chính (self-play).")
    args = parser.parse_args()

    os.makedirs(PHASE4_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    train_ds = AuxeticDataset(os.path.join(PHASE3_DIR, "train.npz"))
    val_ds = AuxeticDataset(os.path.join(PHASE3_DIR, "val.npz"))
    base_n_seeds = train_ds.n_seeds
    base_seed_classes = train_ds.seed_classes
    if args.limit:
        from torch.utils.data import Subset
        train_ds_full, val_ds_full = train_ds, val_ds
        train_ds = Subset(train_ds_full, range(min(args.limit, len(train_ds_full))))
        val_ds = Subset(val_ds_full, range(min(args.limit // 4 or 1, len(val_ds_full))))
        train_ds.n_seeds = train_ds_full.n_seeds
        train_ds.seed_classes = train_ds_full.seed_classes
    if args.adversarial_npz:
        from torch.utils.data import ConcatDataset
        extra = []
        for p in args.adversarial_npz:
            ds = AuxeticDataset(p)
            assert list(ds.seed_classes) == list(base_seed_classes), (
                f"seed_classes của {p} ({list(ds.seed_classes)}) không khớp "
                f"train.npz ({list(base_seed_classes)}) - cột one-hot sẽ lệch."
            )
            extra.extend([ds] * args.adversarial_oversample)
            print(f"  + {len(ds)} mẫu đối kháng từ {p} (x{args.adversarial_oversample} "
                  f"oversample = {len(ds) * args.adversarial_oversample} mẫu hiệu dụng)")
        train_ds = ConcatDataset([train_ds] + extra)
        train_ds.n_seeds = base_n_seeds
        train_ds.seed_classes = base_seed_classes
    print(f"Train: {len(train_ds)} mẫu | Val: {len(val_ds)} mẫu")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    model = SurrogateCNN(n_seeds=train_ds.n_seeds).to(device)
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"Đã load trọng số khởi tạo từ {args.init_from} "
              f"(epoch={init_ckpt.get('epoch')}, val_loss={init_ckpt.get('val_loss')}) "
              f"- fine-tune tiếp thay vì train từ đầu.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(PHASE4_DIR, args.output_name)
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_per_target = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_per_target = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} | "
              f"val_mse[v12,v21,vf]={val_per_target.tolist()} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mse_v12": val_per_target[0].item(),
            "val_mse_v21": val_per_target[1].item(),
            "val_mse_volfrac": val_per_target[2].item(),
        })

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_seeds": train_ds.n_seeds,
                "seed_classes": train_ds.seed_classes.tolist(),
                "channels": (32, 64, 128, 256),
                "fc_hidden": 128,
                "target_names": ["v12", "v21", "volfrac_achieved"],
                "val_loss": val_loss,
                "epoch": epoch,
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping tại epoch {epoch} "
                      f"(không cải thiện val loss trong {args.patience} epoch liên tiếp)")
                break

    import json
    history_name = ("train_history.json" if args.output_name == "surrogate_best.pt"
                     else args.output_name.replace(".pt", "_history.json"))
    history_path = os.path.join(PHASE4_DIR, history_name)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nĐã lưu model tốt nhất: {best_ckpt_path} (val_loss={best_val_loss:.5f})")
    print(f"Lịch sử training: {history_path}")


if __name__ == "__main__":
    main()