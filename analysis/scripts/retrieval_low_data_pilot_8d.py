"""
Fast pilot: retrieval accuracy o 8D khi thu nho tap tra cuu (Ndb)
====================================================================
Nhom 4.2 (docs/archive/PROJECT_PLAN.md) - cong hoi: retrieval chi manh vi co
57.216 mau tra cuu; khi Ndb nho (mo phong mien vat ly moi/khan hiem du
lieu), R2(FE) cua retrieval o khong gian 8D se tut den dau? Day la ban
RE (retrieval-only, khong train lai cVAE) de quyet dinh co dang dau tu
ban CONG BANG (train lai cVAE tren cung Ndb, hang gio GPU) hay khong.

cVAE giu nguyen checkpoint 8D hien tai (cvae_a4_full8d_finetuned_v2.pt,
R2=0.997 lam moc THAM CHIEU, KHONG train lai gi trong script nay).

Thiet ke: CUNG 24 target (seed=123) da dung o moi pilot Nhom 4.2 khac.
Voi moi Ndb trong [500, 1000, 5000, 10000, 68286(full)], lay ngau nhien
(seed co dinh, KHAC seed chon target, tranh trung) Ndb mau tu train.npz
lam "database" tra cuu, tinh lai nearest-neighbor 8D z-score (chuan hoa
theo CHINH Ndb mau do, khong phai toan bo train - dung ngu nghia "chi co
Ndb mau" that su).

Cach chay (can conda env 'simp'):
    /home/tbm/miniconda3/envs/simp/bin/python3 \\
        analysis/scripts/retrieval_low_data_pilot_8d.py

Output: outputs/phase5/reports/retrieval_low_data_pilot_8d.json
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, os.path.dirname(__file__))

from baseline_comparison import _hit_rate_and_r2  # noqa: E402
from curse_of_dimensionality_comparison import _select_targets  # noqa: E402

PHASE3_A4_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_a4")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")

FEATURE_NAMES = ["v12", "v21", "volfrac", "void_size_frac", "nu0"]
N_DB_LEVELS = [500, 1000, 5000, 10000]
SUBSAMPLE_SEED = 999  # khac seed=123 dung de chon target, tranh nham lan


def _retrieval_r2_at_ndb(targets, train_arrays, n_db, rng):
    n_total = len(train_arrays["v12"])
    if n_db < n_total:
        sub_idx = rng.choice(n_total, size=n_db, replace=False)
    else:
        sub_idx = np.arange(n_total)
    sub = {k: v[sub_idx] for k, v in train_arrays.items()}

    means = {k: float(sub[k].mean()) for k in FEATURE_NAMES}
    stds = {k: float(sub[k].std()) for k in FEATURE_NAMES}
    sub_z = np.stack(
        [(sub[k] - means[k]) / stds[k] for k in FEATURE_NAMES], axis=1
    )

    targets_v12 = np.array([t["v12"] for t in targets])
    is_aux = targets_v12 < 0
    preds = np.empty(len(targets))
    dists = np.empty(len(targets))
    for i, t in enumerate(targets):
        cond_z = np.array([(t[k] - means[k]) / stds[k] for k in FEATURE_NAMES])
        d2 = ((sub_z - cond_z) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        preds[i] = sub["v12"][j]
        dists[i] = float(np.sqrt(d2[j]))

    out = _hit_rate_and_r2(targets_v12, preds, is_aux)
    out["n_db"] = int(n_db)
    out["mean_condition_dist_z"] = float(dists.mean())
    return out


def main():
    train = np.load(os.path.join(PHASE3_A4_DIR, "train.npz"), allow_pickle=True)
    param_names = list(train["param_names"])
    void_idx = param_names.index("void_size_frac")
    train_arrays = {
        "v12": train["v12"].astype(np.float64),
        "v21": train["v21"].astype(np.float64),
        "nu0": train["nu"].astype(np.float64),
        "volfrac": train["volfrac_achieved"].astype(np.float64),
        "void_size_frac": train["params"][:, void_idx].astype(np.float64),
    }
    n_total = len(train_arrays["v12"])

    targets = _select_targets(24, seed=123)

    levels = [n for n in N_DB_LEVELS if n < n_total] + [n_total]
    results = []
    for n_db in levels:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        r = _retrieval_r2_at_ndb(targets, train_arrays, n_db, rng)
        results.append(r)
        label = "full" if n_db == n_total else str(n_db)
        print(f"n_db={label:<8} R2={r['r2']:.4f}  MAE={r['mae_v12']:.4f}  "
              f"mean_dist_z={r['mean_condition_dist_z']:.4f}")

    out = {
        "n_conditions": 24,
        "target_seed": 123,
        "subsample_seed": SUBSAMPLE_SEED,
        "n_total_train": n_total,
        "cvae_reference_r2_8d": 0.997,
        "cvae_reference_ckpt": "cvae_a4_full8d_finetuned_v2.pt (not rerun here)",
        "results_by_n_db": results,
    }
    out_path = os.path.join(
        PHASE5_DIR, "reports", "retrieval_low_data_pilot_8d.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
