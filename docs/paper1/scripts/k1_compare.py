"""K1 - so refine nhận thức chế tạo (λ=0,1) với refine hiện tại (λ=0).

Ghép cặp theo condition: cùng seed nên cùng condition, cùng 30 z và cùng
ứng viên best-of-30 - khác nhau chỉ ở objective refine. Chấm 5 tiêu chí ghi
trước trong `docs/plan.md` mục K1 trên thiết kế cuối guarded:
  1. chế tạo được (liên thông 4 hướng + nét tối thiểu, ảnh 64²) ≥ 0,60
  2. R²(ν12) ≥ 0,99
  3. R²(ν21) ≥ 0,80
  4. tỉ lệ thiết kế có chạm góc ≤ 0,13
  5. FE/target ≤ 1,1 × tham chiếu
IN100 = kết quả chính (đủ 5 tiêu chí); IN100-C = xác nhận (tiêu chí 1-2).
Kèm CI95 bootstrap ghép cặp cho chênh lệch tỉ lệ chế tạo và mức giảm MAE.

Chạy từ gốc repo:
    python docs/paper1/scripts/k1_compare.py
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from bootstrap_ci import bootstrap_paired_mae_reduction  # noqa: E402
from manufacturability import (  # noqa: E402
    check_connectivity,
    count_corner_contacts,
)

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
SETS = {
    "IN100": (
        "p1_7_c5_in100_bo30_nofp.json",
        "k1/k1_in100_bo30_l0.1.json",
    ),
    "IN100-C": (
        "p1_9_r6_in100c_bo30_nofp.json",
        "k1/k1_in100c_bo30_l0.1.json",
    ),
}
N_BOOT = 10000


def final_designs(pc: list) -> dict:
    """Thiết kế cuối guarded của từng condition + các chỉ số K1.

    Đo lại chế tạo/chạm góc từ ảnh đã lưu (cả file tham chiếu, vốn chưa có
    trường K1) để 2 phía dùng đúng 1 đoạn code đo.

    Args:
        pc: danh sách per_condition của 1 file benchmark (--save-images).

    Returns:
        dict mảng numpy: v12, v21, t12, t21, manuf4, corner (bool), fe.
    """
    out = {k: [] for k in ("v12", "v21", "t12", "t21", "manuf4", "corner")}
    out["fe"] = []
    for c in pc:
        k = "refined" if c["guarded_accept"] else "baseline"
        img = np.asarray(c[f"{k}_image"])
        out["v12"].append(c[f"{k}_v12"])
        out["v21"].append(c[f"{k}_v21"])
        out["t12"].append(c["target_v12"])
        out["t21"].append(c["target_v21"])
        out["manuf4"].append(
            check_connectivity(img, connectivity=4)["manufacturable"]
        )
        out["corner"].append(count_corner_contacts(img) > 0)
        out["fe"].append(c["n_fe_calls_baseline"] + c["n_fe_calls_refine"])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Hệ số xác định R² của pred so với target."""
    return float(
        1
        - ((pred - target) ** 2).sum() / ((target - target.mean()) ** 2).sum()
    )


def paired_diff_ci(a: np.ndarray, b: np.ndarray, seed: int = 0) -> list:
    """CI95 bootstrap ghép cặp của mean(b) − mean(a) (resample condition).

    Args:
        a, b: giá trị từng condition (cùng thứ tự) của 2 phương pháp.
        seed: seed RNG.

    Returns:
        [điểm, cận dưới, cận trên].
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    boot = (b[idx] - a[idx]).mean(axis=1)
    return [
        float(b.mean() - a.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


def summarize(d: dict) -> dict:
    """Các chỉ số tiêu chí K1 của 1 phương pháp trên 1 tập."""
    pair = np.abs(d["v12"] - d["t12"]) + np.abs(d["v21"] - d["t21"])
    return {
        "manuf4": float(d["manuf4"].mean()),
        "r2_v12": r2(d["v12"], d["t12"]),
        "r2_v21": r2(d["v21"], d["t21"]),
        "frac_corner": float(d["corner"].mean()),
        "fe_per_target": float(d["fe"].mean()),
        "mae_v12": float(np.abs(d["v12"] - d["t12"]).mean()),
        "mae_pair": float(pair.mean()),
    }


def main():
    """So 2 phía trên từng tập, chấm tiêu chí, ghi JSON và in bảng."""
    result = {}
    for name, (ref_file, k1_file) in SETS.items():
        with open(os.path.join(PLAN_V3, ref_file)) as f:
            ref = final_designs(json.load(f)["per_condition"])
        with open(os.path.join(PLAN_V3, k1_file)) as f:
            k1 = final_designs(json.load(f)["per_condition"])
        assert np.allclose(ref["t12"], k1["t12"]), "condition không khớp"
        s_ref, s_k1 = summarize(ref), summarize(k1)
        crit = {
            "1_manuf4>=0.60": s_k1["manuf4"] >= 0.60,
            "2_r2_v12>=0.99": s_k1["r2_v12"] >= 0.99,
            "3_r2_v21>=0.80": s_k1["r2_v21"] >= 0.80,
            "4_corner<=0.13": s_k1["frac_corner"] <= 0.13,
            "5_fe<=1.1x": s_k1["fe_per_target"]
            <= 1.1 * s_ref["fe_per_target"],
        }
        err = lambda d, a, t: np.abs(d[a] - d[t])  # noqa: E731
        result[name] = {
            "ref_lambda0": s_ref,
            "k1_lambda0.1": s_k1,
            "criteria": crit,
            "manuf4_diff_ci95": paired_diff_ci(ref["manuf4"], k1["manuf4"]),
            "mae_v12_reduction": bootstrap_paired_mae_reduction(
                err(ref, "v12", "t12"), err(k1, "v12", "t12"), N_BOOT
            ),
            "mae_pair_reduction": bootstrap_paired_mae_reduction(
                err(ref, "v12", "t12") + err(ref, "v21", "t21"),
                err(k1, "v12", "t12") + err(k1, "v21", "t21"),
                N_BOOT,
            ),
        }
        print(f"\n== {name} ==")
        for key in s_ref:
            print(f"  {key:14s} λ=0 {s_ref[key]:.4f}  λ=0,1 {s_k1[key]:.4f}")
        m = result[name]["manuf4_diff_ci95"]
        print(f"  Δ chế tạo {m[0]:+.2f} [CI95 {m[1]:+.2f}; {m[2]:+.2f}]")
        for key in ("mae_v12_reduction", "mae_pair_reduction"):
            r = result[name][key]
            print(
                f"  giảm {key[:-10]:9s} {r['rel_reduction']:+.1%} "
                f"[{r['rel_reduction_ci95_lo']:+.1%}; "
                f"{r['rel_reduction_ci95_hi']:+.1%}]"
            )
        for key, ok in crit.items():
            print(f"  tiêu chí {key:16s} {'ĐẠT' if ok else 'TRƯỢT'}")
    out = os.path.join(PLAN_V3, "k1", "k1_comparison.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1, default=bool)
    print("\nĐã lưu:", out)


if __name__ == "__main__":
    main()
