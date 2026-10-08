"""P1.8 - cVAE vs SIMP hội tụ phân tầng theo tỉ số dị hướng của mục tiêu.

r = max(ν21*/ν12*, ν12*/ν21*) tính trên MỤC TIÊU (biết trước khi giải), tầng
r < 10 và r ≥ 10. Tiêu chí ghi trước (`docs/plan.md` P1.8, 2026-10-07): trên
IN100-C, tầng r < 10, cận dưới CI95 mức giảm sai số cặp VÀ mức giảm MAE ν12
của cVAE so với SIMP best-of-4 `full` đều > 0. IN100 / IN100-B được tính
cùng cách nhưng chỉ là chẩn đoán hậu kiểm (đã xem số trước khi phân tầng).

Chạy từ gốc repo:
    python docs/paper1/scripts/p1_8_anisotropy_strata.py
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

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
SETS = {
    "IN100 (post-hoc)": (
        "p1_6c_in100_bo30_images.json",
        "p1_6a_simp_baseline_in100.json",
    ),
    "IN100-B (post-hoc)": (
        "p1_7_in100b_bo30_images.json",
        "p1_7_simp_in100b_full.json",
    ),
    "IN100-C (pre-registered)": (
        "p1_8_in100c_bo30_images.json",
        "p1_8_simp_in100c_full.json",
    ),
}
R_SPLIT = 10.0


def load_pair(cvae_file: str, simp_file: str):
    """Target, dự đoán cVAE guarded, dự đoán SIMP `full` best-of-4.

    Returns:
        (T, G, S) mảng (n, 2), đã kiểm target 2 file khớp theo thứ tự.
    """
    with open(os.path.join(PLAN_V3, cvae_file)) as f:
        cv = json.load(f)["per_condition"]
    with open(os.path.join(PLAN_V3, simp_file)) as f:
        sp = json.load(f)["per_condition"]
    assert len(cv) == len(sp)
    assert all(
        abs(a["target_v12"] - b["target_v12"]) < 1e-6 for a, b in zip(cv, sp)
    )
    T = np.array([[c["target_v12"], c["target_v21"]] for c in cv])
    G = np.array(
        [
            (
                [c["refined_v12"], c["refined_v21"]]
                if c["guarded_accept"]
                else [c["baseline_v12"], c["baseline_v21"]]
            )
            for c in cv
        ]
    )
    S = np.array(
        [[s["full"]["best"]["v12"], s["full"]["best"]["v21"]] for s in sp]
    )
    return T, G, S


def strata(T: np.ndarray, G: np.ndarray, S: np.ndarray) -> dict:
    """Số liệu mỗi tầng + CI ghép cặp cVAE vs SIMP (dương = cVAE tốt hơn)."""
    r = np.maximum(T[:, 1] / T[:, 0], T[:, 0] / T[:, 1])
    out = {}
    for name, mask in (("r<10", r < R_SPLIT), ("r>=10", r >= R_SPLIT)):
        eG, eS = np.abs(G - T)[mask], np.abs(S - T)[mask]
        d = {
            "n": int(mask.sum()),
            "cvae_pair_mae": float(eG.sum(1).mean()) if mask.any() else None,
            "simp_pair_mae": float(eS.sum(1).mean()) if mask.any() else None,
            "per_target_pair": (
                [[float(a), float(b)] for a, b in zip(eG.sum(1), eS.sum(1))]
                if name == "r>=10"
                else None
            ),
        }
        if mask.sum() >= 10:
            d["pair"] = bootstrap_paired_mae_reduction(eS.sum(1), eG.sum(1))
            d["v12"] = bootstrap_paired_mae_reduction(eS[:, 0], eG[:, 0])
        out[name] = d
    return out


def main():
    """Tính phân tầng cho mọi tập có đủ file, chấm tiêu chí IN100-C."""
    result = {}
    for name, (cf, sf) in SETS.items():
        if not all(os.path.exists(os.path.join(PLAN_V3, x)) for x in (cf, sf)):
            print(f"{name}: thiếu file, bỏ qua")
            continue
        result[name] = strata(*load_pair(cf, sf))
        s = result[name]["r<10"]
        print(
            f"{name}: r<10 n={s['n']} pair "
            f"{s['pair']['rel_reduction']:+.3f} "
            f"[{s['pair']['rel_reduction_ci95_lo']:+.3f}, "
            f"{s['pair']['rel_reduction_ci95_hi']:+.3f}] | v12 "
            f"{s['v12']['rel_reduction']:+.3f} "
            f"[{s['v12']['rel_reduction_ci95_lo']:+.3f}, "
            f"{s['v12']['rel_reduction_ci95_hi']:+.3f}] | r>=10 n="
            f"{result[name]['r>=10']['n']} "
            f"{result[name]['r>=10']['per_target_pair']}"
        )
    key = "IN100-C (pre-registered)"
    if key in result:
        s = result[key]["r<10"]
        passed = (
            s["pair"]["rel_reduction_ci95_lo"] > 0
            and s["v12"]["rel_reduction_ci95_lo"] > 0
        )
        result["p1_8_pass"] = bool(passed)
        print("P1.8 ĐẠT" if passed else "P1.8 KHÔNG ĐẠT")
    path = os.path.join(PLAN_V3, "p1_8_anisotropy_strata.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
