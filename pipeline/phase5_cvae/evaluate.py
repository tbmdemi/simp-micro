"""
Phase 5 - evaluate.py
=============================================
3 phép kiểm tra latent space sau khi train xong:

1. property_accuracy(): sinh geometry theo condition (v12,v21) của mỗi mẫu
   test, dùng surrogate frozen dự đoán lại Poisson ratio -> R2/MAE.
   CẢNH BÁO: con số này KHÔNG đáng tin - đo hoàn toàn qua surrogate mà
   chính gamma đang tối ưu chống lại, xem outputs/phase5/fe_verification_report.json
   (verify_fe.py) và mục Phase 5 trong README trước khi dùng số ở đây.

2. diversity_check(): giữ condition cố định, sample nhiều z -> đo độ đa
   dạng hình học (pixel-wise std). Gần 0 -> nghi posterior collapse.

3. interpolation(): nội suy tuyến tính z giữa 2 mẫu test -> lưu chuỗi ảnh
   để kiểm tra mắt thường latent space có mượt không.

Cách chạy:
    python3 pipeline/phase5_cvae/evaluate.py

Output:
    outputs/phase5/evaluation_report.json
    outputs/phase5/diagnostics/diversity_condition_X.png
    outputs/phase5/diagnostics/interpolation_XX.png
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import CVAE                     # noqa: E402
from dataset import CVAEDataset            # noqa: E402
from losses import load_frozen_surrogate   # noqa: E402
from sample import load_model, CKPT_PATH   # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
DIAG_DIR = os.path.join(PHASE5_DIR, "diagnostics")


def property_accuracy(model, surrogate, target_names, test_loader, device):
    preds, targets = [], []
    for image, condition, seed_vec, _vf in test_loader:
        condition = condition.to(device)
        seed_vec = seed_vec.to(device)
        with torch.no_grad():
            z = torch.randn(condition.size(0), model.latent_dim, device=device)
            recon = model.decoder(z, condition)
            pred = surrogate(recon, seed_vec)
        idx_v12, idx_v21 = target_names.index("v12"), target_names.index("v21")
        pred_cond = torch.stack([pred[:, idx_v12], pred[:, idx_v21]], dim=1)
        preds.append(pred_cond.cpu().numpy())
        targets.append(condition.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mae = np.abs(preds - targets).mean(axis=0)
    ss_res = ((targets - preds) ** 2).sum(axis=0)
    ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot
    return {
        "v12": {"mae": float(mae[0]), "r2": float(r2[0])},
        "v21": {"mae": float(mae[1]), "r2": float(r2[1])},
        "n_samples": int(len(preds)),
    }


def diversity_check(model, condition, n_samples, device):
    samples = model.generate(
        torch.tensor(condition, dtype=torch.float32, device=device),
        n_samples=n_samples, device=device,
    )  # (n, 1, 64, 64)
    pixel_std = samples.std(dim=0).mean().item()  # trung bình std theo pixel qua n mẫu

    grid = (samples.squeeze(1).cpu().numpy() * 255).astype(np.uint8)  # (n,64,64)
    strip = np.concatenate(list(grid), axis=1)  # ghép ngang thành 1 ảnh dài
    os.makedirs(DIAG_DIR, exist_ok=True)
    fname = f"diversity_v12_{condition[0]:.2f}_v21_{condition[1]:.2f}.png"
    Image.fromarray(strip, mode="L").save(os.path.join(DIAG_DIR, fname))
    return {"condition": condition, "pixel_std": pixel_std, "preview": fname}


def interpolation(model, ds, device, n_steps=8, idx_a=0, idx_b=1):
    img_a, cond_a, seed_a, _ = ds[idx_a]
    img_b, _cond_b, _seed_b, _ = ds[idx_b]
    img_a, img_b = img_a.unsqueeze(0).to(device), img_b.unsqueeze(0).to(device)
    cond_a = cond_a.unsqueeze(0).to(device)

    with torch.no_grad():
        mu_a, _ = model.encoder(img_a, cond_a)
        mu_b, _ = model.encoder(img_b, cond_a)  # dùng chung condition của mẫu A
        frames = []
        for t in np.linspace(0, 1, n_steps):
            z = (1 - t) * mu_a + t * mu_b
            recon = model.decoder(z, cond_a)
            frames.append((recon.squeeze().cpu().numpy() * 255).astype(np.uint8))

    strip = np.concatenate(frames, axis=1)
    os.makedirs(DIAG_DIR, exist_ok=True)
    fname = f"interpolation_{idx_a}_to_{idx_b}.png"
    Image.fromarray(strip, mode="L").save(os.path.join(DIAG_DIR, fname))
    return {"idx_a": idx_a, "idx_b": idx_b, "preview": fname}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Không tìm thấy {CKPT_PATH} - hãy chạy train.py trước.")

    model = load_model(device=device)
    surrogate, target_names = load_frozen_surrogate(device=device)

    test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("1/3 - Đánh giá property accuracy trên test set...")
    prop_report = property_accuracy(model, surrogate, target_names, test_loader, device)
    print(f"   v12: R2={prop_report['v12']['r2']:.4f} MAE={prop_report['v12']['mae']:.4f}")
    print(f"   v21: R2={prop_report['v21']['r2']:.4f} MAE={prop_report['v21']['mae']:.4f}")

    print("2/3 - Kiểm tra đa dạng hình học (condition cố định)...")
    diversity_report = diversity_check(
        model, condition=[-0.6, -0.6], n_samples=8, device=device
    )
    print(f"   pixel_std={diversity_report['pixel_std']:.4f} "
          f"(gần 0 -> nghi ngờ posterior collapse)")

    print("3/3 - Nội suy latent space giữa 2 mẫu test...")
    interp_report = interpolation(model, test_ds, device, n_steps=8, idx_a=0, idx_b=1)

    report = {
        "property_accuracy": prop_report,
        "diversity_check": diversity_report,
        "interpolation": interp_report,
    }
    os.makedirs(PHASE5_DIR, exist_ok=True)
    with open(os.path.join(PHASE5_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nĐã lưu báo cáo: outputs/phase5/evaluation_report.json")
    print(f"Ảnh chẩn đoán: outputs/phase5/diagnostics/")


if __name__ == "__main__":
    main()