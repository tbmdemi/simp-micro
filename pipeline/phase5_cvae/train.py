"""
Phase 5 - train.py
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
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import CVAE                     # noqa: E402
from dataset import CVAEDataset            # noqa: E402
from losses import (                       # noqa: E402
    cvae_loss, kl_beta_schedule, load_frozen_surrogate,
    load_frozen_surrogate_ensemble, prior_sample_regularization,
)
from verify_fe import (                    # noqa: E402
    FE_PARAMS, resize_to_fe_grid, evaluate_density_field,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


def real_fe_r2(model, val_conditions: np.ndarray, device) -> float:
    """R2(v12) đo bằng FE THẬT (không qua surrogate) trên 1 tập condition cố
    định - dùng làm tín hiệu chọn checkpoint trong training loop (Phần 3 kế
    hoạch self-play), thay vì chỉ báo cáo hậu-kỳ như verify_fe.py gốc. Tái
    dùng đúng FE_PARAMS/resize_to_fe_grid/evaluate_density_field đã
    sanity-check ở verify_fe.py, không viết lại logic FE."""
    model.eval()
    targets, preds = [], []
    for cond in val_conditions:
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        with torch.no_grad():
            img = model.generate(cond_t, n_samples=1, device=device)
        img64 = img.squeeze().cpu().numpy().astype(np.float32)
        img_bin = (img64 > 0.5).astype(np.float32)
        img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
        try:
            v12_fe, _v21_fe, _ = evaluate_density_field(img_fe, FE_PARAMS)
        except Exception:
            continue
        targets.append(cond[0])
        preds.append(v12_fe)
    if len(targets) < 2:
        return float("nan")
    targets = np.array(targets)
    preds = np.array(preds)
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def run_epoch(model, loader, surrogate, target_names, optimizer, beta, gamma,
              lambda_tv, lambda_bin, device, train: bool, lambda_disagreement=0.0,
              lambda_periodic=0.0, regularize_prior_samples=False):
    model.train(mode=train)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "prop": 0.0,
              "prop_weighted": 0.0, "tv": 0.0, "binarization": 0.0,
              "periodic": 0.0, "disagreement": 0.0,
              "prior_tv": 0.0, "prior_binarization": 0.0, "prior_periodic": 0.0}
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
                lambda_disagreement=lambda_disagreement,
                lambda_periodic=lambda_periodic,
            )
            if regularize_prior_samples:
                prior_reg_total, prior_stats = prior_sample_regularization(
                    model.decoder, model.latent_dim, condition,
                    lambda_tv=lambda_tv, lambda_bin=lambda_bin,
                    lambda_periodic=lambda_periodic,
                )
                losses["total"] = losses["total"] + prior_reg_total
                losses.update(prior_stats)
            else:
                losses.update({"prior_tv": torch.tensor(0.0), "prior_binarization": torch.tensor(0.0),
                                "prior_periodic": torch.tensor(0.0)})
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
        totals["periodic"] += losses["periodic"].item() * bsz
        totals["disagreement"] += float(losses["disagreement"]) * bsz
        totals["prior_tv"] += losses["prior_tv"].item() * bsz
        totals["prior_binarization"] += losses["prior_binarization"].item() * bsz
        totals["prior_periodic"] += losses["prior_periodic"].item() * bsz
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
    parser.add_argument("--lambda-periodic", type=float, default=0.0,
                         help="Roadmap 6.3: trọng số periodicity loss (MSE giữa cột "
                              "trái/phải + hàng trên/dưới ảnh reconstruct - xem "
                              "losses.periodicity_loss, manufacturability.check_periodicity). "
                              "Mặc định 0.0 (tắt) để giữ tương thích baseline cũ.")
    parser.add_argument("--regularize-prior-samples", action="store_true",
                         help="ÁP THÊM lambda-tv/lambda-bin/lambda-periodic lên ảnh decode "
                              "từ z ~ PRIOR N(0,1) (cùng chế độ model.generate() dùng lúc "
                              "inference), KHÔNG chỉ trên `recon` posterior (qua encoder) "
                              "như mặc định - xem losses.prior_sample_regularization "
                              "docstring: recon posterior đã gần-tuần hoàn sẵn (ảnh training "
                              "thật), nên regularize nó không cải thiện manufacturability lúc "
                              "generate(); cần regularize đúng chế độ prior mới có tác dụng.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5,
                         help="lr tối thiểu ở cuối CosineAnnealing (0 = tắt schedule, giữ lr cố định)")
    parser.add_argument("--patience", type=int, default=15,
                         help="early stopping: dừng nếu val loss không giảm sau N epoch")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--surrogate-path", type=str, default=None,
                         help="Đường dẫn surrogate_for_phase5.pt khác mặc định "
                              "(vd checkpoint đã fine-tune đối kháng ở 1 vòng self-play). "
                              "Bỏ trống = dùng SURROGATE_PATH mặc định trong losses.py. "
                              "Bỏ qua nếu dùng --surrogate-paths (ensemble).")
    parser.add_argument("--surrogate-paths", type=str, nargs="*", default=None,
                         help="2+ đường dẫn surrogate_for_phase5.pt độc lập (vd huấn "
                              "luyện với khởi tạo/seed khác nhau) - dùng ENSEMBLE thay "
                              "vì 1 surrogate đông cứng: property loss = MSE(trung bình "
                              "dự đoán, target) + lambda-disagreement * phương sai giữa "
                              "các surrogate. Biện pháp cấu trúc chống exploitation, "
                              "khó đánh lừa đồng thời N mô hình độc lập hơn 1 mô hình.")
    parser.add_argument("--lambda-disagreement", type=float, default=0.0,
                         help="Trọng số phạt phương sai giữa các surrogate trong "
                              "ensemble (chỉ có tác dụng khi dùng --surrogate-paths). "
                              "0.0 = chỉ dùng trung bình, không phạt bất đồng.")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Checkpoint cVAE (.pt) để load model_state_dict trước khi "
                              "train tiếp, thay vì khởi tạo ngẫu nhiên (self-play).")
    parser.add_argument("--fe-eval-every", type=int, default=0,
                         help="Cứ mỗi N epoch, chạy FE THẬT (không qua surrogate) trên 1 "
                              "tập condition validation cố định, log R2(FE) vào history. "
                              "0 = tắt (mặc định, giữ nguyên hành vi cũ).")
    parser.add_argument("--select-by", choices=["val_loss", "fe_r2"], default="val_loss",
                         help="Tiêu chí chọn checkpoint tốt nhất. 'fe_r2' cần "
                              "--fe-eval-every > 0 - chọn theo R2(FE thật) thay vì val_loss "
                              "(vốn có thể bị surrogate exploitation đánh lừa, xem README §5).")
    parser.add_argument("--n-fe-eval-conditions", type=int, default=8,
                         help="Số condition validation dùng cho --fe-eval-every.")
    parser.add_argument("--output-name", type=str, default="cvae_best.pt",
                         help="Tên checkpoint lưu trong outputs/phase5/ - đổi tên này để "
                              "không ghi đè cvae_best.pt chính (self-play).")
    args = parser.parse_args()

    if args.select_by == "fe_r2" and args.fe_eval_every <= 0:
        raise ValueError("--select-by fe_r2 cần --fe-eval-every > 0.")

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
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(f"Đã load trọng số từ {args.resume_from} "
              f"(epoch={resume_ckpt.get('epoch')}, val_loss={resume_ckpt.get('val_loss')}) "
              f"- train tiếp thay vì khởi tạo ngẫu nhiên.")
    if args.surrogate_paths:
        assert len(args.surrogate_paths) >= 2, (
            "--surrogate-paths cần >= 2 checkpoint để tạo ensemble có ý nghĩa "
            "(1 checkpoint thì dùng --surrogate-path thay vào)."
        )
        surrogate, target_names = load_frozen_surrogate_ensemble(
            args.surrogate_paths, device=device
        )
        print(f"Ensemble surrogate: {len(surrogate)} model độc lập "
              f"({args.surrogate_paths}), lambda_disagreement={args.lambda_disagreement}")
    elif args.surrogate_path:
        surrogate, target_names = load_frozen_surrogate(device=device, path=args.surrogate_path)
    else:
        surrogate, target_names = load_frozen_surrogate(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_min > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )

    fe_eval_conditions = None
    if args.fe_eval_every > 0:
        rng = np.random.default_rng(42)
        n_cond = min(args.n_fe_eval_conditions, len(val_ds))
        idxs = rng.choice(len(val_ds), size=n_cond, replace=False)
        fe_eval_conditions = np.stack(
            [[val_ds[i][1][0].item(), val_ds[i][1][1].item()] for i in idxs]
        )
        print(f"FE-eval bật: mỗi {args.fe_eval_every} epoch chấm R2(FE thật) trên "
              f"{n_cond} condition validation cố định.")

    history = []
    best_val = float("-inf") if args.select_by == "fe_r2" else float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        beta = kl_beta_schedule(epoch, args.kl_warmup, beta_max=1.0)

        train_stats = run_epoch(
            model, train_loader, surrogate, target_names,
            optimizer, beta, args.gamma * (epoch / 20 if epoch < 20 else 1),
            args.lambda_tv, args.lambda_bin, device, train=True,
            lambda_disagreement=args.lambda_disagreement,
            lambda_periodic=args.lambda_periodic,
            regularize_prior_samples=args.regularize_prior_samples,
        )
        val_stats = run_epoch(
            model, val_loader, surrogate, target_names,
            optimizer, beta, args.gamma * (epoch / 20 if epoch < 20 else 1),
            args.lambda_tv, args.lambda_bin, device, train=False,
            lambda_disagreement=args.lambda_disagreement,
            lambda_periodic=args.lambda_periodic,
            regularize_prior_samples=args.regularize_prior_samples,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(f"[{epoch:03d}/{args.epochs}] beta={beta:.3f} lr={current_lr:.2e} | "
              f"train total={train_stats['total']:.2f} recon={train_stats['recon']:.2f} "
              f"kl={train_stats['kl']:.3f} prop={train_stats['prop']:.4f} "
              f"prop_w={train_stats['prop_weighted']:.2f} tv={train_stats['tv']:.4f} "
              f"bin={train_stats['binarization']:.4f} periodic={train_stats['periodic']:.4f} "
              f"prior[tv={train_stats['prior_tv']:.4f} bin={train_stats['prior_binarization']:.4f} "
              f"periodic={train_stats['prior_periodic']:.4f}] "
              f"disagree={train_stats['disagreement']:.5f} || "
              f"val total={val_stats['total']:.2f} recon={val_stats['recon']:.2f} "
              f"kl={val_stats['kl']:.3f} prop={val_stats['prop']:.4f} "
              f"prop_w={val_stats['prop_weighted']:.2f} tv={val_stats['tv']:.4f} "
              f"bin={val_stats['binarization']:.4f} periodic={val_stats['periodic']:.4f} "
              f"prior[tv={val_stats['prior_tv']:.4f} bin={val_stats['prior_binarization']:.4f} "
              f"periodic={val_stats['prior_periodic']:.4f}] "
              f"disagree={val_stats['disagreement']:.5f}")

        fe_r2 = None
        if fe_eval_conditions is not None and epoch % args.fe_eval_every == 0:
            fe_r2 = real_fe_r2(model, fe_eval_conditions, device)
            print(f"    [FE-eval epoch {epoch}] R2(v12, FE thật)={fe_r2:.4f}")

        history.append({"epoch": epoch, "beta": beta,
                         "train": train_stats, "val": val_stats, "fe_r2": fe_r2})

        ckpt_path = os.path.join(PHASE5_DIR, args.output_name)
        if args.select_by == "fe_r2":
            if fe_r2 is not None and not np.isnan(fe_r2) and fe_r2 > best_val:
                best_val = fe_r2
                epochs_no_improve = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "latent_dim": args.latent_dim,
                    "condition_dim": 2,
                    "resolution": args.resolution,
                    "epoch": epoch,
                    "val_loss": val_stats["total"],
                    "fe_r2": best_val,
                    "gamma": args.gamma,
                    "lambda_tv": args.lambda_tv,
                    "lambda_bin": args.lambda_bin,
                    "lambda_periodic": args.lambda_periodic,
                }, ckpt_path)
            elif fe_r2 is not None:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping tại epoch {epoch} "
                          f"(R2(FE thật) không tăng trong {args.patience} lần FE-eval)")
                    break
        else:
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
                    "fe_r2": fe_r2,
                    "gamma": args.gamma,
                    "lambda_tv": args.lambda_tv,
                    "lambda_bin": args.lambda_bin,
                    "lambda_periodic": args.lambda_periodic,
                }, ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping tại epoch {epoch} "
                          f"(val loss không giảm trong {args.patience} epoch)")
                    break

    history_name = ("train_history.json" if args.output_name == "cvae_best.pt"
                     else args.output_name.replace(".pt", "_history.json"))
    with open(os.path.join(PHASE5_DIR, history_name), "w") as f:
        json.dump(history, f, indent=2)

    metric_name = "R2(FE)" if args.select_by == "fe_r2" else "val_loss"
    print(f"Đã lưu checkpoint tốt nhất: outputs/phase5/{args.output_name} "
          f"({metric_name}={best_val:.4f})")


if __name__ == "__main__":
    main()