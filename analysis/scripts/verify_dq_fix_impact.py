"""
Kiểm chứng tác động THẬT của bug-fix dQ (2026-07-24, xem EXPERIMENT_LOG.md
mục "Các Lỗi Chính đã Sửa" và README.md Known Limitations #10) lên nghiệm
topology của 3 seed BẤT ĐỐI XỨNG (hourglass, hexagonal, reentrant_bowtie) -
những seed duy nhất mà lỗi hoán vị dQ có thể làm sai hướng gradient OC (8/11
seed còn lại đối xứng qua đường chéo nên transpose vô hại, xem docstring
simp/homogenization/compute.py).

Chạy 2 lần SIMP THẬT (không mock) với CÙNG params/seed - 1 lần dùng code
hiện tại (đã sửa), 1 lần dùng lại đúng công thức dQ CŨ (buggy, order='C',
tái tạo tại chỗ trong file này - KHÔNG sửa lại simp/homogenization/compute.py)
- rồi so sánh v12/v21/manufacturability/số vòng lặp hội tụ, trên NHIỀU bộ
tham số DOE (LHS trong đúng khoảng đã narrow ở batch 8 - xem
outputs/multi_batch/batch_8/batch_8_params.json), không chỉ 1 điểm đại diện
như lần chạy đầu (2026-07-24) - xem EXPERIMENT_LOG.md để biết kết quả lần đó.

Cách chạy:
    python3 analysis/scripts/verify_dq_fix_impact.py                  # N_PARAM_COMBOS mặc định (4)
    python3 analysis/scripts/verify_dq_fix_impact.py --n-combos 8     # nhiều bộ tham số hơn
    python3 analysis/scripts/verify_dq_fix_impact.py --n-workers 10   # chạy song song (khuyến nghị)

Thời gian ước tính: 3 seed x N_PARAM_COMBOS x 2 phiên bản dQ lần chạy SIMP
đầy đủ trên lưới 50x50, mỗi lần vài chục giây (max_iter=200) -> N_PARAM_COMBOS=4
mất khoảng 8-12 phút TUẦN TỰ (--n-workers 0, mặc định), hoặc ~1-2 phút với
--n-workers 10 (mỗi (seed, combo, phiên bản) hoàn toàn độc lập, xem
pipeline/phase5_cvae/real_physics.py cho cùng pattern multiprocessing.Pool).

Output: in bảng so sánh + tổng hợp theo seed ra terminal, lưu
outputs/dq_fix_verification/impact_report.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from unittest.mock import patch

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from simp.runner import run_simp                                            # noqa: E402
from pipeline.phase5_cvae.manufacturability import check_manufacturability  # noqa: E402

AFFECTED_SEEDS = ["hourglass", "hexagonal", "reentrant_bowtie"]

# Khoảng tham số đã narrow ở batch 8 (nguồn: chính xác từ
# outputs/multi_batch/batch_8/batch_8_params.json, không phải ước lượng lại).
PARAM_RANGES = {
    "volfrac": (0.49980250000000004, 0.58491925),
    "penal": (2.56433175, 3.658632),
    "rmin": (1.05755775, 1.24652425),
    "move": (0.0747135, 0.14423875),
    "void_size_frac": (0.283641, 0.381436),
}

FIXED_PARAMS = dict(nelx=50, nely=50, max_iter=200, verbose=False, save_every=200)

OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "dq_fix_verification")


def _sample_param_combos(n: int, seed: int = 42) -> list:
    """LHS đơn giản (numpy, không phụ thuộc scipy.stats.qmc) trong PARAM_RANGES -
    n bộ tham số DOE, dùng CHUNG cho cả 3 seed (cô lập đúng biến seed/dQ)."""
    rng = np.random.default_rng(seed)
    names = list(PARAM_RANGES.keys())
    combos = []
    # LHS thủ công: chia [0,1) thành n khoảng đều, xáo trộn độc lập từng chiều.
    strata = (np.arange(n) + rng.random((len(names), n))) / n
    for i in range(len(names)):
        rng.shuffle(strata[i])
    for j in range(n):
        combo = {}
        for i, name in enumerate(names):
            lo, hi = PARAM_RANGES[name]
            combo[name] = float(lo + strata[i, j] * (hi - lo))
        combos.append(combo)
    return combos


def _buggy_compute_homogenized_tensor(U, U0, xPhys, KE, edofMat, penal, E0, Emin, rho0=1.0):
    """Tái tạo NGUYÊN VĂN compute_homogenized_tensor() TRƯỚC bug-fix
    2026-07-24 (reshape order='C' thay vì 'F') - CHỈ dùng để so sánh trong
    script này, không phải bản để dùng lại nơi khác."""
    nely, nelx = xPhys.shape
    nele = nelx * nely
    edofMat_0 = edofMat - 1
    Ue = np.zeros((nele, 8, 3))
    for i in range(nele):
        for j in range(3):
            Ue[i, :, j] = U[edofMat_0[i, :], j]
    x_flat = xPhys.flatten('F')
    E_penal = Emin + (rho0 * x_flat ** penal) * (E0 - Emin)
    k_e = E_penal
    Q = np.einsum('e,emi,mn,enj->ij', k_e, Ue, KE, Ue) / (nelx * nely)
    dk_e = rho0 * penal * x_flat ** (penal - 1) * (E0 - Emin)
    dQ_flat = np.einsum('e,emi,mn,enj->eij', dk_e, Ue, KE, Ue) / (nelx * nely)
    dQ = dQ_flat.transpose(1, 2, 0).reshape(3, 3, nely, nelx)  # BUG: thiếu order='F'
    return Q, dQ, Ue


def _run_one(seed_name: str, combo_idx: int, combo_params: dict, use_buggy_dq: bool) -> dict:
    tag = "buggy" if use_buggy_dq else "fixed"
    output_dir = os.path.join(OUTPUT_ROOT, f"{seed_name}_combo{combo_idx}_{tag}")
    params = dict(FIXED_PARAMS, **combo_params, seed=seed_name, output_dir=output_dir)

    if use_buggy_dq:
        with patch("simp.runner.compute_homogenized_tensor", _buggy_compute_homogenized_tensor):
            result = run_simp(params)
    else:
        result = run_simp(params)

    xPhys_bin = (result["xPhys"] > 0.5).astype(np.float32)
    manuf = check_manufacturability(xPhys_bin)

    return {
        "seed": seed_name,
        "combo_idx": combo_idx,
        "dq_version": tag,
        "v12": result["v12"],
        "v21": result["v21"],
        "n_iters": result["n_iters"],
        "converged": bool(result["converged"]),
        "objective": result["objective"],
        # "passes_all" = liên thông AND tuần hoàn (kiểm tra tổng hợp - xem
        # check_manufacturability() docstring); "manufacturable" trong
        # manuf chỉ là kết quả riêng của check_connectivity(), KHÔNG phải
        # tổng hợp - dùng passes_all để so sánh cho đúng convention với
        # pipeline/phase2_multi_batch/runner.py.
        "passes_all": bool(manuf["passes_all"]),
        "is_connected": bool(manuf["is_connected"]),
        "periodic_ok": bool(manuf["periodic_ok"]),
    }


def _run_task(task: tuple) -> dict:
    """Hàm top-level (bắt buộc để pickle được cho multiprocessing.Pool) - unwrap
    1 task (seed, combo_idx, combo_params, use_buggy_dq) rồi gọi _run_one()."""
    seed_name, combo_idx, combo_params, use_buggy_dq = task
    return _run_one(seed_name, combo_idx, combo_params, use_buggy_dq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-combos", type=int, default=4,
                         help="Số bộ tham số DOE (LHS trong khoảng batch 8) thử cho mỗi seed.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed cho LHS (tái lập được).")
    parser.add_argument("--n-workers", type=int, default=0,
                         help="Số tiến trình song song (multiprocessing.Pool). "
                              "0 = tuần tự (mặc định). Mỗi (seed, combo, phiên bản dQ) "
                              "hoàn toàn độc lập nên song song gần như tuyến tính theo core.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    combos = _sample_param_combos(args.n_combos, seed=args.seed)
    print(f"Đã sinh {len(combos)} bộ tham số DOE (LHS, seed={args.seed}):")
    for i, c in enumerate(combos):
        print(f"  combo {i}: " + ", ".join(f"{k}={v:.4f}" for k, v in c.items()))

    tasks = [
        (seed_name, i, combo, use_buggy_dq)
        for seed_name in AFFECTED_SEEDS
        for i, combo in enumerate(combos)
        for use_buggy_dq in (False, True)
    ]

    if args.n_workers > 0:
        print(f"\nChạy song song ({args.n_workers} tiến trình, {len(tasks)} task)...")
        with mp.Pool(args.n_workers) as pool:
            task_results = pool.map(_run_task, tasks)
    else:
        task_results = [_run_task(t) for t in tasks]

    # Ghép fixed/buggy lại theo (seed, combo_idx) - thứ tự tasks đảm bảo fixed
    # luôn đứng trước buggy cho cùng (seed, combo_idx).
    by_key = {}
    for r in task_results:
        by_key.setdefault((r["seed"], r["combo_idx"]), {})[r["dq_version"]] = r

    rows = []
    for seed_name in AFFECTED_SEEDS:
        print(f"\n=== seed={seed_name} ===")
        for i, combo in enumerate(combos):
            fixed = by_key[(seed_name, i)]["fixed"]
            buggy = by_key[(seed_name, i)]["buggy"]
            delta = fixed["v12"] - buggy["v12"]
            flipped = fixed["passes_all"] != buggy["passes_all"]
            sign_flip = (fixed["v12"] < 0) != (buggy["v12"] < 0)
            print(f"  combo {i}: [fixed] v12={fixed['v12']:+.4f}  [buggy] v12={buggy['v12']:+.4f}  "
                  f"delta={delta:+.4f}  sign_flip={sign_flip}  manuf_flip={flipped}")
            rows.append({"seed": seed_name, "combo_idx": i, "combo_params": combo,
                          "fixed": fixed, "buggy": buggy, "delta_v12": delta,
                          "sign_flipped": sign_flip, "manufacturability_flipped": flipped})

    print("\n=== TỔNG HỢP THEO SEED ===")
    print(f"{'seed':22s} {'n_combo':>8s} {'mean_delta':>11s} {'sign_flip_rate':>15s} {'manuf_flip_rate':>16s}")
    summary = {}
    for seed_name in AFFECTED_SEEDS:
        seed_rows = [r for r in rows if r["seed"] == seed_name]
        mean_delta = float(np.mean([r["delta_v12"] for r in seed_rows]))
        sign_flip_rate = float(np.mean([r["sign_flipped"] for r in seed_rows]))
        manuf_flip_rate = float(np.mean([r["manufacturability_flipped"] for r in seed_rows]))
        summary[seed_name] = {"mean_delta_v12": mean_delta, "sign_flip_rate": sign_flip_rate,
                               "manuf_flip_rate": manuf_flip_rate, "n_combos": len(seed_rows)}
        print(f"{seed_name:22s} {len(seed_rows):>8d} {mean_delta:>11.4f} "
              f"{sign_flip_rate:>15.2%} {manuf_flip_rate:>16.2%}")

    report_path = os.path.join(OUTPUT_ROOT, "impact_report.json")
    with open(report_path, "w") as f:
        json.dump({"rows": rows, "summary": summary, "n_combos": args.n_combos,
                    "lhs_seed": args.seed}, f, indent=2)
    print(f"\nĐã lưu: {report_path}")
    print(
        "\nCách đọc: sign_flip_rate cao (vd > 30-50%) nghĩa là bug thường xuyên đổi "
        "kết quả từ auxetic sang không-auxetic (hoặc ngược lại) trên KHẮP không gian "
        "tham số, không chỉ 1 điểm - bằng chứng mạnh cho việc cần rebuild dataset của "
        "seed đó. sign_flip_rate thấp nhưng mean_delta lớn nghĩa là bug làm auxeticity "
        "yếu đi đáng kể dù không đổi dấu hẳn - vẫn đáng cân nhắc rebuild. Cả 2 chỉ số "
        "thấp mới có thể coi bug ít ảnh hưởng thực nghiệm với seed đó."
    )


if __name__ == "__main__":
    main()
