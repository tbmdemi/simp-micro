"""
Phase 4 - evaluate.py  (roadmap bước 4.3 + 4.4)
==================================================
Đánh giá surrogate model đã train trên test.npz:
  - R², MAE riêng cho từng target (v12, v21, volfrac_achieved)
  - Sai số theo TỪNG SEED (phát hiện seed khó như reentrant_bowtie/hexagonal)
  - Sai số theo BIN giá trị v12 (phát hiện vùng biên/vùng thưa dữ liệu)

Cách chạy:
    python3 pipeline/phase4_surrogate/evaluate.py
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AuxeticDataset
from model import SurrogateCNN

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE4_DIR = os.path.join(REPO_ROOT, "outputs", "phase4")


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ckpt_path = os.path.join(PHASE4_DIR, "surrogate_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateCNN(
        n_seeds=ckpt["n_seeds"], channels=ckpt["channels"], fc_hidden=ckpt["fc_hidden"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = AuxeticDataset(os.path.join(PHASE3_DIR, "test.npz"))
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    preds, targets_all, seed_names_all = [], [], []
    with torch.no_grad():
        for image, seed_vec, targets in loader:
            pred = model(image.to(device), seed_vec.to(device)).cpu().numpy()
            preds.append(pred)
            targets_all.append(targets.numpy())
    preds = np.concatenate(preds)
    targets_all = np.concatenate(targets_all)
    seed_names_all = test_ds.seed_onehot.argmax(axis=1)
    seed_classes = test_ds.seed_classes

    target_names = ckpt["target_names"]
    report = {"overall": {}, "per_seed": {}, "per_v12_bin": {}}

    print("=" * 60)
    print("KẾT QUẢ TỔNG QUÁT (bước 4.3)")
    print("=" * 60)
    for i, name in enumerate(target_names):
        y_true, y_pred = targets_all[:, i], preds[:, i]
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        report["overall"][name] = {"r2": float(r2), "mae": float(mae)}
        # Ngưỡng roadmap 4.3 chỉ đặt ra cho v12/v21 (>=0.90); volfrac_achieved
        # dùng ngưỡng lỏng hơn (>=0.80) vì nó là target phụ trợ.
        threshold = 0.90 if name in ("v12", "v21") else 0.80
        status = "ĐẠT" if r2 >= threshold else f"CHƯA ĐẠT (mục tiêu R2>={threshold})"
        print(f"  {name:20s}  R2={r2:.4f}  MAE={mae:.4f}   [{status}]")

    print("\n" + "=" * 60)
    print("SAI SỐ THEO TỪNG SEED (bước 4.4 - phát hiện seed khó)")
    print("=" * 60)
    print(f"  {'seed':22s} {'n':>5s} {'MAE_v12':>10s} {'MAE_v21':>10s}")
    for s_idx, s_name in enumerate(seed_classes):
        mask = seed_names_all == s_idx
        if mask.sum() == 0:
            continue
        mae_v12 = np.mean(np.abs(targets_all[mask, 0] - preds[mask, 0]))
        mae_v21 = np.mean(np.abs(targets_all[mask, 1] - preds[mask, 1]))
        report["per_seed"][str(s_name)] = {
            "n": int(mask.sum()), "mae_v12": float(mae_v12), "mae_v21": float(mae_v21)
        }
        print(f"  {s_name:22s} {mask.sum():5d} {mae_v12:10.4f} {mae_v21:10.4f}")

    print("\n" + "=" * 60)
    print("SAI SỐ THEO BIN v12 (bước 4.4 - phát hiện vùng biên/thưa)")
    print("=" * 60)
    bins = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.5]
    v12_true = targets_all[:, 0]
    print(f"  {'khoảng v12':>18s} {'n':>6s} {'MAE_v12':>10s}")
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (v12_true >= b0) & (v12_true < b1)
        if mask.sum() == 0:
            continue
        mae_bin = np.mean(np.abs(v12_true[mask] - preds[mask, 0]))
        report["per_v12_bin"][f"[{b0},{b1})"] = {"n": int(mask.sum()), "mae_v12": float(mae_bin)}
        print(f"  [{b0:5.2f}, {b1:5.2f})  {mask.sum():6d} {mae_bin:10.4f}")

    report_path = os.path.join(PHASE4_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo: {report_path}")


if __name__ == "__main__":
    main()