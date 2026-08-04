"""
OOD baseline comparison: cVAE vs nearest-neighbor retrieval NGOAI PHAN PHOI
============================================================================
baseline_comparison.py da tra loi "trong phan phoi cVAE co thang retrieval
khong" - CHUA (retrieval R2=1.000 vs cVAE best-of-10 R2=0.973, n=24) - vi
dataset qua day (57k mau) nen retrieval luon tim duoc mau train rat gan
target (mean_condition_dist~0.002). Cau hoi con treo (EXPERIMENT_LOG.md
2026-08-02, LIMITATIONS.md #18): "vay ngoai-phan-phoi thi sao?"

Thiet ke target OOD (kiem tra truc tiep tren train.npz, xem
EXPERIMENT_LOG.md muc OOD 2026-08-04):
  - v12/v21 TOAN BO 57,216 mau train deu AM (auxetic) - min=-1.9527,
    max=-0.002314. KHONG CO mau nao v12>0 (non-auxetic/Poisson ratio thuong)
    trong dataset - day la vung OOD "cung" nhat co the chon: retrieval
    KHONG THE tra ve gia tri duong vi khong ton tai trong train.npz.
  - "extreme_auxetic": v12 vuot qua min train (-1.9527) - vd -2.1..-2.8 -
    ngoai pham vi da thay, doi hoi ngoai suy ve cuong do auxetic.
  - "non_auxetic" (sign-flip): v12 duong (0.1..0.8) - vung mat do训练=0
    tuyet doi, retrieval bi ep phai tra ve mau AM gan 0 nhat (sai dau hoan
    toan), con cVAE co the (hoac khong the) ngoai suy dung dau.

Voi moi target, chay nearest_neighbor_baseline (tai su dung tu
baseline_comparison.py) VA best_of_n() (custom_condition, tai su dung tu
best_of_n_eval.py, KHONG viet lai) tren CUNG 1 checkpoint.

Cach chay:
    python3 analysis/scripts/ood_baseline_comparison.py \\
        --cvae-ckpt outputs/phase5/cvae_realphysics.pt --cvae-n-samples 10

Output: outputs/phase5/reports/ood_baseline_comparison_<ckpt-stem>.json
"""
import os
import sys
import json
import argparse

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, os.path.dirname(__file__))

from best_of_n_eval import best_of_n              # noqa: E402
from baseline_comparison import nearest_neighbor_baseline, _hit_rate_and_r2  # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")

# (v12, v21, category) - xem rationale trong docstring o tren. v21=v12 cho
# don gian (train co corr(v12,v21)~0.05, gan nhu doc lap - khong co huong
# "tu nhien" nao de suy ra v21 tu v12, nen chon duong cheo doi xung la lua
# chon trung lap, de giai thich nhat).
OOD_TARGETS = [
    (-2.10, -2.10, "extreme_auxetic"),
    (-2.30, -2.30, "extreme_auxetic"),
    (-2.50, -2.50, "extreme_auxetic"),
    (-2.80, -2.80, "extreme_auxetic"),
    (-2.20, -1.60, "extreme_auxetic"),
    (-2.60, -1.90, "extreme_auxetic"),
    (0.10, 0.10, "non_auxetic"),
    (0.30, 0.30, "non_auxetic"),
    (0.50, 0.50, "non_auxetic"),
    (0.80, 0.80, "non_auxetic"),
    (0.20, 0.50, "non_auxetic"),
    (0.50, 0.20, "non_auxetic"),
]


def run(cvae_ckpt: str, cvae_n_samples: int, device: str, seed: int = 123) -> dict:
    train = np.load(os.path.join(PHASE3_DIR, "train.npz"), allow_pickle=True)
    train_v12 = train["v12"].astype(np.float64)
    train_v21 = train["v21"].astype(np.float64)

    conditions = [np.array([v12, v21], dtype=np.float64) for v12, v21, _ in OOD_TARGETS]
    categories = [c for _, _, c in OOD_TARGETS]

    nn_res = nearest_neighbor_baseline(conditions, train_v12, train_v21)
    for pc, cat in zip(nn_res["per_condition"], categories):
        pc["category"] = cat

    cvae_per_condition = []
    for v12, v21, cat in OOD_TARGETS:
        single = best_of_n(cvae_ckpt, n_conditions=1, n_samples=cvae_n_samples,
                            device=device, seed=seed,
                            custom_condition=np.array([v12, v21], dtype=np.float32))
        if single["per_condition"]:
            entry = dict(single["per_condition"][0])
        else:
            entry = {"target_v12": v12, "v12_best": float("nan"), "v12_first": float("nan"),
                     "n_valid_samples": 0}
        entry["category"] = cat
        entry["target_v21"] = v21
        cvae_per_condition.append(entry)

    targets_v12 = np.array([e["target_v12"] for e in cvae_per_condition])
    is_aux = targets_v12 < 0
    preds_best = np.array([e["v12_best"] for e in cvae_per_condition])
    preds_first = np.array([e["v12_first"] for e in cvae_per_condition])
    cvae_res_best = _hit_rate_and_r2(targets_v12, preds_best, is_aux)
    cvae_res_first = _hit_rate_and_r2(targets_v12, preds_first, is_aux)

    def _by_category(preds):
        out = {}
        for cat in ("extreme_auxetic", "non_auxetic"):
            idx = [i for i, c in enumerate(categories) if c == cat]
            t = targets_v12[idx]
            p = preds[idx]
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            mae = float(np.abs(t - p).mean())
            out[cat] = {"r2": r2, "mae_v12": mae, "n": len(idx)}
        return out

    return {
        "n_conditions": len(OOD_TARGETS),
        "seed": seed,
        "cvae_ckpt": cvae_ckpt,
        "cvae_n_samples": cvae_n_samples,
        "train_v12_range": [float(train_v12.min()), float(train_v12.max())],
        "ood_targets": [{"v12": v12, "v21": v21, "category": cat} for v12, v21, cat in OOD_TARGETS],
        "nearest_neighbor_retrieval": nn_res,
        "nearest_neighbor_by_category": _by_category(np.array(
            [pc["pred_v12"] for pc in nn_res["per_condition"]])),
        "cvae_best_of_n": cvae_res_best,
        "cvae_single_shot": cvae_res_first,
        "cvae_best_of_n_by_category": _by_category(preds_best),
        "cvae_single_shot_by_category": _by_category(preds_first),
        "cvae_per_condition": cvae_per_condition,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae-ckpt", type=str,
                         default=os.path.join(PHASE5_DIR, "cvae_realphysics.pt"))
    parser.add_argument("--cvae-n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run(args.cvae_ckpt, args.cvae_n_samples, device, seed=args.seed)

    ckpt_stem = os.path.splitext(os.path.basename(args.cvae_ckpt))[0]
    out_path = args.out or os.path.join(PHASE5_DIR, "reports", f"ood_baseline_comparison_{ckpt_stem}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"N OOD targets = {result['n_conditions']} (train v12 range = {result['train_v12_range']})")
    print(f"{'Phuong an':<32}{'R2 (all)':>10}{'MAE(v12)':>10}")
    n = result["nearest_neighbor_retrieval"]
    c = result["cvae_best_of_n"]
    print(f"{'nearest-neighbor retrieval':<32}{n['r2']:>10.3f}{n['mae_v12']:>10.4f}")
    print(f"{'cVAE best-of-' + str(args.cvae_n_samples):<32}{c['r2']:>10.3f}{c['mae_v12']:>10.4f}")
    print("\nTheo tung nhom:")
    for cat in ("extreme_auxetic", "non_auxetic"):
        nn_c = result["nearest_neighbor_by_category"][cat]
        cv_c = result["cvae_best_of_n_by_category"][cat]
        print(f"  {cat:<18} retrieval R2={nn_c['r2']:>7.3f} MAE={nn_c['mae_v12']:.4f}"
              f"   |   cVAE R2={cv_c['r2']:>7.3f} MAE={cv_c['mae_v12']:.4f}")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
