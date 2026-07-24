"""
Phase 4 - bootstrap_ci.py
============================================================
evaluate.py bao cao R2/MAE cua surrogate tren test.npz (1.184 mau) nhu 1
diem uoc luong don, khong co khoang tin cay. Khac voi Phase 5 (n=19-24 dieu
kien, CI rat rong - xem pipeline/phase5_cvae/bootstrap_ci.py), test set o
day lon (1.184 mau doc lap) nen ky vong CI hep hon nhieu - script nay do
that de kiem chung thay vi gia dinh.

KHONG huan luyen lai (khong can k-fold CV ton kem) - chi chay 1 lan forward
pass tren checkpoint da co (`surrogate_best.pt`) + test.npz, roi percentile
bootstrap tren cac mau ca nhan (resample tung sample, khac Phase 5 resample
tung dieu kien).

Cach chay:
    python3 pipeline/phase4_surrogate/bootstrap_ci.py
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AuxeticDataset                              # noqa: E402
from model import SurrogateCNN                                   # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE4_DIR = os.path.join(REPO_ROOT, "outputs", "phase4")

TARGET_NAMES = ["v12", "v21", "volfrac_achieved"]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    if np.ptp(y_true) < 1e-9:
        return float("nan")
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def bootstrap_r2(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 10000,
                  seed: int = 0) -> dict:
    n = len(y_true)
    point = r2_score(y_true, y_pred)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = r2_score(y_true[idx], y_pred[idx])
    valid = boot[~np.isnan(boot)]
    lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (float("nan"), float("nan"))
    return {
        "n_samples": int(n),
        "r2_point_estimate": point,
        "r2_ci95_lo": float(lo),
        "r2_ci95_hi": float(hi),
        "n_boot_valid": int(len(valid)),
        "n_boot_total": int(n_boot),
    }


def get_test_predictions(ckpt_path: str, test_npz_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SurrogateCNN(
        n_seeds=ckpt["n_seeds"], channels=ckpt["channels"], fc_hidden=ckpt["fc_hidden"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = AuxeticDataset(test_npz_path)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    preds, targets_all = [], []
    with torch.no_grad():
        for image, seed_vec, targets in loader:
            pred = model(image.to(device), seed_vec.to(device)).cpu().numpy()
            preds.append(pred)
            targets_all.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(targets_all)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str,
                         default=os.path.join(PHASE4_DIR, "surrogate_best.pt"))
    parser.add_argument("--test-npz", type=str,
                         default=os.path.join(PHASE3_DIR, "test.npz"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                         default=os.path.join(PHASE4_DIR, "bootstrap_ci_report.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds, targets_all = get_test_predictions(args.ckpt, args.test_npz, device)

    report = {}
    for i, name in enumerate(TARGET_NAMES):
        stats = bootstrap_r2(targets_all[:, i], preds[:, i], n_boot=args.n_boot, seed=args.seed)
        report[name] = stats
        print(f"{name:20s}  R2 = {stats['r2_point_estimate']:.4f}   "
              f"95% CI (bootstrap, n={stats['n_samples']} mau) = "
              f"[{stats['r2_ci95_lo']:.4f}, {stats['r2_ci95_hi']:.4f}]")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDa ghi {args.out}")


if __name__ == "__main__":
    main()
