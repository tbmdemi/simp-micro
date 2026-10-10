"""
Phase 5 - benchmark_hybrid.py
============================================================
Thí nghiệm lai cVAE → SIMP (plan.md v3, P1.7 - tiêu chí ghi 2026-10-06
TRƯỚC khi viết code/chạy; KHÔNG đổi β, ngân sách hay ngưỡng sau khi có số).

Điểm khởi tạo SIMP = thiết kế cuối cVAE best-of-30 + refine guarded (ảnh
nhị phân lưu bởi `benchmark_physics_guided_refinement.py --save-images`, đã
qua force_periodic → ngưỡng 0,5), resize nearest về lưới FE 50² đúng như
verify, clip [X_MIN, 1] + co về volfrac_max nếu vượt (cùng quy tắc
`benchmark_simp_baseline.initial_design`). SIMP giữ nguyên objective, ràng
buộc và đường verify của `benchmark_simp_baseline.py`, chỉ đổi:
  * β bắt đầu sắc để giữ tô-pô khởi tạo: β ∈ {8, 16, 32, 64},
  * tối đa 15 eval/mức β (≤ 61 FE gồm verify).

Chọn kết quả cuối kiểu guarded: giữ thiết kế lai chỉ khi sai số cặp verify
|Δν12|+|Δν21| nhỏ hơn thiết kế cVAE khởi tạo (không tốn FE thêm).

4 tiêu chí (so SIMP best-of-4 `full`, đủ 100 target, giữ cả #46):
  1. cận dưới CI95 mức giảm sai số cặp ghép cặp ≥ −10% (không-kém-hơn),
  2. tỉ lệ chế tạo được (passes_all trên ảnh nhị phân cuối) ≥ 0,70,
  3. FE trung bình/target (cVAE + lai) ≤ 299,
  4. tỉ lệ thiết kế có E_x hoặc E_y < 1e-3·E0 (lưới 50²) ≤ 1%.

Cách chạy:
    python pipeline/phase5_cvae/benchmark_hybrid.py \
        --cvae-json outputs/phase5/plan_v3/p1_6c_in100_bo30_images.json \
        --simp-json outputs/phase5/plan_v3/p1_6a_simp_baseline_in100.json \
        --out outputs/phase5/plan_v3/p1_7_hybrid_in100.json
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

import benchmark_simp_baseline as sb  # noqa: E402
from bootstrap_ci import bootstrap_paired_mae_reduction  # noqa: E402
from manufacturability import check_manufacturability  # noqa: E402
from verify_fe import (  # noqa: E402
    FE_PARAMS,
    evaluate_density_field,
    resize_to_fe_grid,
)

from simp.objectives.auxetic import compute_elastic_constants  # noqa: E402

HYBRID_BETAS = (8.0, 16.0, 32.0, 64.0)
HYBRID_EVALS_PER_BETA = 15
CRITERIA = {
    "pair_rel_reduction_ci_lo_min": -0.10,
    "frac_manufacturable_min": 0.70,
    "mean_fe_max": 299.0,
    "frac_degenerate_max": 0.01,
}
DEGENERATE_E = 1e-3  # E_x hoặc E_y < 1e-3·E0 → ô gần đứt (như #46)


def cvae_start(cond: dict) -> dict:
    """Thiết kế cVAE cuối (guarded) của 1 condition: ảnh 50² + số verify.

    Args:
        cond: 1 phần tử per_condition của JSON refine có ảnh.

    Returns:
        dict: design50 (nhị phân, 50×50), v12, v21, manufacturable, n_fe.
    """
    key = "refined" if cond["guarded_accept"] else "baseline"
    img = np.asarray(cond[f"{key}_image"], dtype=np.float32)
    return {
        "design50": resize_to_fe_grid(img, 50, 50),
        "v12": cond[f"{key}_v12"],
        "v21": cond[f"{key}_v21"],
        "manufacturable": bool(cond[f"{key}_manufacturable"]),
        "n_fe": int(cond["n_fe_calls_baseline"] + cond["n_fe_calls_refine"]),
    }


def feasible_x0(design50: np.ndarray) -> np.ndarray:
    """Ảnh nhị phân 50² → điểm khởi tạo MMA khả thi (flatten 'F').

    Cùng quy tắc `sb.initial_design`: co về volfrac_max nếu đặc hơn, clip
    [X_MIN, 1] (nlopt từ chối x0 vi phạm bound).
    """
    x0 = design50.astype(np.float64)
    vmax = sb.SIMP_PARAMS["volfrac_max"]
    if x0.mean() > vmax:
        x0 = x0 * vmax / x0.mean()
    return np.clip(x0, sb.X_MIN, 1.0).flatten("F")


def moduli(design50: np.ndarray) -> tuple:
    """(E_x/E0, E_y/E0) của ảnh nhị phân 50² (0, 0 nếu FE lỗi)."""
    try:
        _, _, Q = evaluate_density_field(design50, FE_PARAMS)
        ec = compute_elastic_constants(Q)
        return ec["E_x"] / FE_PARAMS["E0"], ec["E_y"] / FE_PARAMS["E0"]
    except Exception:
        return 0.0, 0.0


def _job(args):
    """(chỉ số, target, ảnh cVAE 50²) → kết quả SIMP lai + mô-đun."""
    i, target, design50 = args
    res = sb.run_simp_target(
        target,
        "cvae",
        HYBRID_EVALS_PER_BETA,
        x0=feasible_x0(design50),
        betas=HYBRID_BETAS,
    )
    img = np.asarray(res["image"], dtype=np.float64)
    res["E_x"], res["E_y"] = moduli(img)
    return i, res


def select_guarded(start: dict, hybrid: dict, target) -> dict:
    """Giữ thiết kế lai chỉ khi sai số cặp verify nhỏ hơn thiết kế cVAE.

    Returns:
        dict: v12, v21, manufacturable, E_x, E_y, accepted (bool).
    """
    e_start = sb._pair_err(start["v12"], start["v21"], target)
    e_hyb = sb._pair_err(hybrid["v12"], hybrid["v21"], target)
    if e_hyb < e_start:
        src, ok = hybrid, True
    else:
        src, ok = start, False
    return {
        "v12": src["v12"],
        "v21": src["v21"],
        "manufacturable": bool(src["manufacturable"]),
        "E_x": src["E_x"],
        "E_y": src["E_y"],
        "accepted": ok,
    }


def evaluate_criteria(final: list, simp_full: list, targets, fe: list):
    """Chấm 4 tiêu chí đặt trước (xem docstring module).

    Args:
        final: kết quả guarded mỗi condition (select_guarded).
        simp_full: SIMP best-of-4 `full` mỗi condition (dict v12/v21).
        targets: list (ν12*, ν21*).
        fe: FE/target (cVAE + lai).

    Returns:
        dict số liệu + cờ đạt từng tiêu chí + "all_pass".
    """
    err_h = [
        sb._pair_err(f["v12"], f["v21"], t) for f, t in zip(final, targets)
    ]
    err_s = [
        sb._pair_err(s["v12"], s["v21"], t) for s, t in zip(simp_full, targets)
    ]
    ci = bootstrap_paired_mae_reduction(err_s, err_h)
    manuf = float(np.mean([f["manufacturable"] for f in final]))
    degen = float(
        np.mean([min(f["E_x"], f["E_y"]) < DEGENERATE_E for f in final])
    )
    mean_fe = float(np.mean(fe))
    out = {
        "pair_mae_hybrid": float(np.mean(err_h)),
        "pair_mae_simp_full": float(np.mean(err_s)),
        "pair_vs_simp_full": ci,
        "frac_manufacturable": manuf,
        "mean_fe": mean_fe,
        "frac_degenerate": degen,
        "frac_hybrid_accepted": float(np.mean([f["accepted"] for f in final])),
    }
    out["pass"] = {
        "c1_pair_noninferior": ci["rel_reduction_ci95_lo"]
        >= CRITERIA["pair_rel_reduction_ci_lo_min"],
        "c2_manufacturable": manuf >= CRITERIA["frac_manufacturable_min"],
        "c3_cost": mean_fe <= CRITERIA["mean_fe_max"],
        "c4_nondegenerate": degen <= CRITERIA["frac_degenerate_max"],
    }
    out["all_pass"] = all(out["pass"].values())
    return out


def main():
    """CLI: chạy lai cho mọi condition, chấm tiêu chí, ghi JSON."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--cvae-json", required=True)
    p.add_argument("--simp-json", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.cvae_json) as f:
        cv = json.load(f)["per_condition"]
    targets = [(c["target_v12"], c["target_v21"]) for c in cv]
    starts = [cvae_start(c) for c in cv]
    for s in starts:
        s["E_x"], s["E_y"] = moduli(s["design50"])

    t0 = time.time()
    hybrids = [None] * len(cv)
    jobs = [
        (i, t, s["design50"]) for i, (t, s) in enumerate(zip(targets, starts))
    ]
    with Pool(args.workers) as pool:
        for i, res in pool.imap_unordered(_job, jobs):
            hybrids[i] = res
    print(f"Lai xong {len(cv)} condition sau {time.time() - t0:.0f} s")

    final = [
        select_guarded(s, h, t) for s, h, t in zip(starts, hybrids, targets)
    ]
    fe = [s["n_fe"] + h["n_fe"] for s, h in zip(starts, hybrids)]
    per_condition = [
        {
            "target_v12": t[0],
            "target_v21": t[1],
            "cvae": {k: v for k, v in s.items() if k != "design50"},
            "hybrid": h,
            "final": f,
            "n_fe_total": n,
        }
        for t, s, h, f, n in zip(targets, starts, hybrids, final, fe)
    ]
    out = {
        "config": {
            "cvae_json": args.cvae_json,
            "simp_json": args.simp_json,
            "betas": list(HYBRID_BETAS),
            "evals_per_beta": HYBRID_EVALS_PER_BETA,
            "criteria": CRITERIA,
        },
        "per_condition": per_condition,
    }
    if args.simp_json:
        with open(args.simp_json) as f:
            sp = json.load(f)["per_condition"]
        assert all(
            abs(a["target_v12"] - b["target_v12"]) < 1e-9
            for a, b in zip(cv, sp)
        )
        out["criteria_eval"] = evaluate_criteria(
            final, [c["full"]["best"] for c in sp], targets, fe
        )
        print(json.dumps(out["criteria_eval"], indent=1, default=float))
    with open(args.out, "w") as f:
        json.dump(out, f, default=float)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
