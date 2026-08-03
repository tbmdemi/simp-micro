"""
Baseline comparison: cVAE vs random vs nearest-neighbor retrieval
============================================================
Trả lời trực tiếp audit report (AUDIT_REPORT_SCIENTIFIC_REVIEW_2026-08-02.md
cau hoi #2 va #15): "Ban dang so sanh voi cai gi?" / "Neu khong co cVAE thi
ket qua con giu duoc khong?" - pipeline hien khong co baseline nao ngoai so
sanh optimizer noi bo (OC vs MMA), khong co so sanh voi "khong dung mo hinh
sinh" o muc do bai toan inverse design.

3 phuong an, CUNG mot tap target condition (seed=123, lay tu test.npz -
GIONG HET cach best_of_n_eval.py chon, de so sanh apples-to-apples):

1. random: voi moi target, lay 1 anh NGAU NHIEN tu train.npz (bo qua
   target), dung nhan (v12,v21) THAT cua no (da co san, khong can chay lai
   FE) - san sang do "may man mu".
2. nearest_neighbor: voi moi target, tim mau trong train.npz co (v12,v21)
   THAT gan target nhat (khoang cach Euclid trong khong gian 2D thuoc
   tinh) - mo phong "neu khong co mo hinh sinh, chi tra cuu dataset da co
   san thi duoc bao nhieu". Day la baseline chuan trong tai lieu inverse-
   design/generative model (vd so sanh voi retrieval-based baseline).
3. cvae: goi thang best_of_n_eval.best_of_n() (khong viet lai) tren CUNG
   condition set.

GIOI HAN THUC SU (khong che giau): khong co baseline "SIMP-only nham vao
1 target Poisson ratio cu the" vi simp/objectives/auxetic.py hien CHI co
muc tieu cuc tri hoa (minimize Q12), KHONG co so hang bam muc tieu
(vd (v12-target)^2) - can code them 1 objective moi de lam duoc dieu nay,
ngoai pham vi lan nay. Xem "known_gap" trong output.

Cach chay:
    python3 analysis/scripts/baseline_comparison.py --n-conditions 24 --cvae-n-samples 10

Output: outputs/phase5/reports/baseline_comparison_<ckpt-stem>.json
"""
import os
import sys
import json
import argparse

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from dataset import CVAEDataset       # noqa: E402
from best_of_n_eval import best_of_n  # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


def _select_conditions(n_conditions: int, seed: int):
    """Y het logic chon condition trong best_of_n_eval.best_of_n() khi
    custom_condition=None - dam bao 2 ham dung CHUNG 1 tap target."""
    test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
    return [test_ds[i][1].numpy() for i in idxs]


def _hit_rate_and_r2(targets_v12: np.ndarray, preds_v12: np.ndarray, is_auxetic_target: np.ndarray) -> dict:
    n_auxetic = int(is_auxetic_target.sum())
    n_hits = int(((preds_v12 < 0) & is_auxetic_target).sum())
    hit_rate = n_hits / n_auxetic if n_auxetic else float("nan")
    ss_res = ((targets_v12 - preds_v12) ** 2).sum()
    ss_tot = ((targets_v12 - targets_v12.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.abs(targets_v12 - preds_v12).mean())
    return {"hit_rate": hit_rate, "r2": r2, "mae_v12": mae, "n_auxetic_targets": n_auxetic}


def random_baseline(conditions: list, train_v12: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    targets = np.array([c[0] for c in conditions])
    is_aux = targets < 0
    picks = rng.integers(0, len(train_v12), size=len(conditions))
    preds = train_v12[picks]
    out = _hit_rate_and_r2(targets, preds, is_aux)
    out["per_condition"] = [
        {"target_v12": float(t), "pred_v12": float(p)} for t, p in zip(targets, preds)
    ]
    return out


def nearest_neighbor_baseline(conditions: list, train_v12: np.ndarray, train_v21: np.ndarray) -> dict:
    targets = np.array([c[0] for c in conditions])
    is_aux = targets < 0
    preds = np.empty(len(conditions), dtype=np.float64)
    dists = np.empty(len(conditions), dtype=np.float64)
    for i, cond in enumerate(conditions):
        d2 = (train_v12 - cond[0]) ** 2 + (train_v21 - cond[1]) ** 2
        j = int(np.argmin(d2))
        preds[i] = train_v12[j]
        dists[i] = float(np.sqrt(d2[j]))
    out = _hit_rate_and_r2(targets, preds, is_aux)
    out["per_condition"] = [
        {"target_v12": float(t), "pred_v12": float(p), "condition_dist": float(d)}
        for t, p, d in zip(targets, preds, dists)
    ]
    out["mean_condition_dist"] = float(dists.mean())
    return out


def run(n_conditions: int, cvae_ckpt: str, cvae_n_samples: int, device: str, seed: int = 123) -> dict:
    train = np.load(os.path.join(PHASE3_DIR, "train.npz"), allow_pickle=True)
    train_v12 = train["v12"].astype(np.float64)
    train_v21 = train["v21"].astype(np.float64)

    conditions = _select_conditions(n_conditions, seed)

    random_res = random_baseline(conditions, train_v12, seed=seed)
    nn_res = nearest_neighbor_baseline(conditions, train_v12, train_v21)
    cvae_res = best_of_n(cvae_ckpt, n_conditions, cvae_n_samples, device, seed=seed)

    cvae_summary = {
        "hit_rate_single_shot": cvae_res["hit_rate_single_shot"],
        "hit_rate_best_of_n": cvae_res["hit_rate_best_of_n"],
        "r2": cvae_res["r2_fe_v12_best_of_n"],
        "n_auxetic_targets": cvae_res["n_auxetic_targets"],
        "n_samples_per_condition": cvae_n_samples,
    }

    return {
        "n_conditions": n_conditions,
        "seed": seed,
        "cvae_ckpt": cvae_ckpt,
        "random": random_res,
        "nearest_neighbor_retrieval": nn_res,
        "cvae_best_of_n": cvae_summary,
        "cvae_full_result": cvae_res,
        "known_gap": (
            "Khong co baseline 'SIMP-only nham vao 1 target Poisson ratio cu "
            "the' - simp/objectives/auxetic.py hien chi cuc tri hoa Q12, "
            "khong co so hang bam muc tieu. Can them objective moi de lam "
            "duoc, ngoai pham vi script nay."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-conditions", type=int, default=24)
    parser.add_argument("--cvae-ckpt", type=str,
                         default=os.path.join(PHASE5_DIR, "cvae_v2_finetuned.pt"))
    parser.add_argument("--cvae-n-samples", type=int, default=10,
                         help="So mau cVAE sinh/condition (FE that tren tat ca, xem best_of_n_eval.py).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run(args.n_conditions, args.cvae_ckpt, args.cvae_n_samples, device, seed=args.seed)

    ckpt_stem = os.path.splitext(os.path.basename(args.cvae_ckpt))[0]
    out_path = args.out or os.path.join(PHASE5_DIR, "reports", f"baseline_comparison_{ckpt_stem}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    r, n, c = result["random"], result["nearest_neighbor_retrieval"], result["cvae_best_of_n"]
    print(f"N condition = {args.n_conditions} (seed={args.seed}, cung tap target ca 3 phuong an)")
    print(f"{'Phuong an':<32}{'hit_rate':>10}{'R2':>10}{'MAE(v12)':>10}")
    print(f"{'random (train ngau nhien)':<32}{r['hit_rate']:>10.3f}{r['r2']:>10.3f}{r['mae_v12']:>10.4f}")
    print(f"{'nearest-neighbor retrieval':<32}{n['hit_rate']:>10.3f}{n['r2']:>10.3f}{n['mae_v12']:>10.4f}"
          f"   (mean_condition_dist={n['mean_condition_dist']:.4f})")
    print(f"{'cVAE single-shot':<32}{c['hit_rate_single_shot']:>10.3f}{'':>10}{'':>10}")
    print(f"{'cVAE best-of-' + str(args.cvae_n_samples):<32}{c['hit_rate_best_of_n']:>10.3f}{c['r2']:>10.3f}")
    print(f"\nGioi han chua giai quyet: {result['known_gap']}")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
