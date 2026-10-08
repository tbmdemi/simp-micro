"""
Curse-of-dimensionality comparison: retrieval vs cVAE khi condition mo rong
tu 2 chieu (v12,v21) len 4 chieu (v12,v21,nu0) len 8 chieu
(v12,v21,volfrac,void_size_frac,nu0)
============================================================================
Nhom 4.2 (docs/archive/PROJECT_PLAN.md muc 3, 4): baseline_comparison.py da chi
ra
retrieval THANG cVAE trong-phan-phoi o condition 2D (R2=1.000 vs 0.973,
LIMITATIONS.md muc 18) - vi dataset qua day (mean_condition_dist=0.002).
Gia thuyet can kiem dinh: khi Giai doan A (A7, 2026-08-19) them nu0 lam
dieu kien thu 3 (condition_dim 2->4 o cvae_a4_nu0_finetuned.pt), khong gian
tro nen thua hon - retrieval mat loi the mat do, con cVAE thi khong (hoc
ham lien tuc, khong phu thuoc mat do local). Day la luan diem trung tam de
tra loi "vi sao can generative thay vi tra bang" - quan trong hon ca thi
nghiem sign-flip OOD da co (ood_baseline_comparison.py, 2026-08-04).

Thiet ke: lay CUNG 1 tap target (v12,v21,nu0) thuc te tu
outputs/phase3_a4/test.npz (khong bia tay nhu OOD script - o day dang do
hieu ung MAT DO trong-phan-phoi, khong phai kha nang ngoai suy, nen target
phai la joint sample THAT, giu dung tuong quan v12/v21/nu0 von co trong du
lieu). Voi CUNG tap target nay, so sanh 2 "khong gian dieu kien":
  - "2D": chi (v12,v21) - checkpoint cvae_a4_control_finetuned.pt
    (condition_dim=2), retrieval tren (train_v12,train_v21).
  - "4D": (v12,v21,nu0) - checkpoint cvae_a4_nu0_finetuned.pt
    (condition_dim=4), retrieval tren (train_v12,train_v21,train_nu).
  - "8D": (v12,v21,volfrac,void_size_frac,nu0) - checkpoint
    cvae_a4_full8d_finetuned.pt (condition_dim=8, train 2-stage 2026-08-20,
    xem EXPERIMENT_LOG.md). volfrac/void_size_frac da co san trong
    outputs/phase3_a4 (volfrac_achieved + params/param_names), khong can
    backfill.

nearest_neighbor_baseline() (2D, tai su dung tu baseline_comparison.py)
dung khoang cach Euclid THO - du vi std(v12)~std(v21)~0.2 nen khong lech
scale. O 4D PHAI chuan hoa z-score truoc khi tinh khoang cach
(nearest_neighbor_baseline_nd() duoi day) vi std(nu0)~0.023 nho hon ~9 lan
std(v12,v21) (outputs/phase3_a4/train.npz, do truc tiep) - neu dung khoang
cach tho, chieu nu0 gan nhu khong anh huong ket qua nearest-neighbor, bien
phep do "curse of dimensionality" thanh phep do bi chi phoi boi scale thay
vi boi so chieu that.

Cach chay (can conda env 'simp', xem README.md):
    /home/tbm/miniconda3/envs/simp/bin/python3 \\
        analysis/scripts/curse_of_dimensionality_comparison.py \\
        --n-conditions 24 --cvae-n-samples 10

Output: outputs/phase5/reports/curse_of_dimensionality_comparison.json
"""

import argparse
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
from best_of_n_eval import best_of_n  # noqa: E402

PHASE3_A4_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_a4")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")

CKPT_2D = os.path.join(PHASE5_DIR, "cvae_a4_control_finetuned.pt")
CKPT_4D = os.path.join(PHASE5_DIR, "cvae_a4_nu0_finetuned.pt")
CKPT_8D = os.path.join(PHASE5_DIR, "cvae_a4_full8d_finetuned_v2.pt")


def _select_targets(n_conditions: int, seed: int):
    """Lay n_conditions bo (v12,v21,nu0,volfrac,void_size_frac) THAT tu
    test.npz (outputs/phase3_a4) - joint sample giu dung tuong quan du lieu,
    khong bia tay nhu OOD script (o day dang do hieu ung MAT DO
    trong-phan-phoi, khong phai OOD)."""
    test = np.load(os.path.join(PHASE3_A4_DIR, "test.npz"), allow_pickle=True)
    param_names = list(test["param_names"])
    void_idx = param_names.index("void_size_frac")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test["v12"]), size=n_conditions, replace=False)
    return [
        {
            "v12": float(test["v12"][i]),
            "v21": float(test["v21"][i]),
            "nu0": float(test["nu"][i]),
            "volfrac": float(test["volfrac_achieved"][i]),
            "void_size_frac": float(test["params"][i, void_idx]),
        }
        for i in idxs
    ]


def nearest_neighbor_baseline_nd(
    targets: list, train_arrays: dict, feature_names: list
) -> dict:
    """Nearest-neighbor retrieval N-chieu, khoang cach Euclid chuan hoa
    z-score (xem rationale ve scale o docstring dau file)."""
    means = {k: float(train_arrays[k].mean()) for k in feature_names}
    stds = {k: float(train_arrays[k].std()) for k in feature_names}
    train_z = np.stack(
        [(train_arrays[k] - means[k]) / stds[k] for k in feature_names], axis=1
    )

    targets_v12 = np.array([t["v12"] for t in targets])
    is_aux = targets_v12 < 0
    preds = np.empty(len(targets))
    dists = np.empty(len(targets))
    for i, t in enumerate(targets):
        cond_z = np.array([(t[k] - means[k]) / stds[k] for k in feature_names])
        d2 = ((train_z - cond_z) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        preds[i] = train_arrays["v12"][j]
        dists[i] = float(np.sqrt(d2[j]))

    out = _hit_rate_and_r2(targets_v12, preds, is_aux)
    out["per_condition"] = [
        {
            "target_v12": float(t["v12"]),
            "pred_v12": float(p),
            "condition_dist_z": float(d),
        }
        for t, p, d in zip(targets, preds, dists)
    ]
    out["mean_condition_dist_z"] = float(dists.mean())
    return out


def _cvae_run(
    ckpt: str,
    targets: list,
    n_samples: int,
    device: str,
    seed: int,
    use_nu0: bool = False,
    use_extended: bool = False,
) -> dict:
    per_condition = []
    for t in targets:
        kwargs = {}
        if use_nu0:
            kwargs["nu0"] = t["nu0"]
        if use_extended:
            kwargs["volfrac"] = t["volfrac"]
            kwargs["void_size_frac"] = t["void_size_frac"]
        single = best_of_n(
            ckpt,
            n_conditions=1,
            n_samples=n_samples,
            device=device,
            seed=seed,
            custom_condition=np.array([t["v12"], t["v21"]], dtype=np.float32),
            **kwargs,
        )
        if single["per_condition"]:
            entry = dict(single["per_condition"][0])
        else:
            entry = {
                "target_v12": t["v12"],
                "v12_best": float("nan"),
                "v12_first": float("nan"),
                "n_valid_samples": 0,
            }
        per_condition.append(entry)

    targets_v12 = np.array([e["target_v12"] for e in per_condition])
    is_aux = targets_v12 < 0
    preds_best = np.array([e["v12_best"] for e in per_condition])
    return {
        "best_of_n": _hit_rate_and_r2(targets_v12, preds_best, is_aux),
        "per_condition": per_condition,
    }


def run(
    n_conditions: int, cvae_n_samples: int, device: str, seed: int = 123
) -> dict:
    train = np.load(
        os.path.join(PHASE3_A4_DIR, "train.npz"), allow_pickle=True
    )
    param_names = list(train["param_names"])
    void_idx = param_names.index("void_size_frac")
    train_arrays = {
        "v12": train["v12"].astype(np.float64),
        "v21": train["v21"].astype(np.float64),
        "nu0": train["nu"].astype(np.float64),
        "volfrac": train["volfrac_achieved"].astype(np.float64),
        "void_size_frac": train["params"][:, void_idx].astype(np.float64),
    }

    targets = _select_targets(n_conditions, seed)

    # --- khong gian 2D (v12,v21) ---
    nn_2d = nearest_neighbor_baseline_nd(targets, train_arrays, ["v12", "v21"])
    cvae_2d = _cvae_run(
        CKPT_2D, targets, n_samples=cvae_n_samples, device=device, seed=seed
    )

    # --- khong gian 4D (v12,v21,nu0) ---
    nn_4d = nearest_neighbor_baseline_nd(
        targets, train_arrays, ["v12", "v21", "nu0"]
    )
    cvae_4d = _cvae_run(
        CKPT_4D,
        targets,
        n_samples=cvae_n_samples,
        device=device,
        seed=seed,
        use_nu0=True,
    )

    # --- khong gian 8D (v12,v21,volfrac,void_size_frac,nu0) ---
    nn_8d = nearest_neighbor_baseline_nd(
        targets,
        train_arrays,
        ["v12", "v21", "volfrac", "void_size_frac", "nu0"],
    )
    cvae_8d = _cvae_run(
        CKPT_8D,
        targets,
        n_samples=cvae_n_samples,
        device=device,
        seed=seed,
        use_nu0=True,
        use_extended=True,
    )

    spaces = {
        "space_2d": (CKPT_2D, nn_2d, cvae_2d),
        "space_4d": (CKPT_4D, nn_4d, cvae_4d),
        "space_8d": (CKPT_8D, nn_8d, cvae_8d),
    }
    return {
        "n_conditions": n_conditions,
        "cvae_n_samples": cvae_n_samples,
        "seed": seed,
        "targets": targets,
        **{
            key: {
                "ckpt": ckpt,
                "nearest_neighbor_retrieval": nn,
                "cvae_best_of_n": cvae["best_of_n"],
                "cvae_per_condition": cvae["per_condition"],
            }
            for key, (ckpt, nn, cvae) in spaces.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-conditions", type=int, default=24)
    parser.add_argument("--cvae-n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run(
        args.n_conditions, args.cvae_n_samples, device, seed=args.seed
    )

    out_path = args.out or os.path.join(
        PHASE5_DIR, "reports", "curse_of_dimensionality_comparison.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"N targets = {result['n_conditions']} (joint samples that tu test.npz)\n"
    )
    print(
        f"{'Khong gian':<8}{'Phuong an':<28}{'R2':>8}{'MAE(v12)':>10}{'mean_dist_z':>13}"
    )
    for space_key, label in (
        ("space_2d", "2D"),
        ("space_4d", "4D"),
        ("space_8d", "8D"),
    ):
        s = result[space_key]
        n, c = s["nearest_neighbor_retrieval"], s["cvae_best_of_n"]
        print(
            f"{label:<8}{'nearest-neighbor retrieval':<28}{n['r2']:>8.3f}"
            f"{n['mae_v12']:>10.4f}{n['mean_condition_dist_z']:>13.4f}"
        )
        print(
            f"{label:<8}{'cVAE best-of-' + str(args.cvae_n_samples):<28}{c['r2']:>8.3f}"
            f"{c['mae_v12']:>10.4f}{'':>13}"
        )
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
