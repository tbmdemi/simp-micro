"""
Pilot có kiểm soát: OC hai-multiplier (stiffness_mode='dual', simp/core/oc.py
::oc_update_dual) so với 'gate' (mặc định hiện tại) trên seed `hexagonal`.

Bối cảnh (xem EXPERIMENT_LOG.md mục 2026-07-29 "ACCEPTED"): sau khi sửa 2 bug
nghiêm trọng (Q-scaling A1 + x=0 absorbing state), yield sạch `hexagonal` rơi
93,9%->80,5%. Ba hướng khắc phục đã thử (tăng delta, tăng beta, damping+
continuation) đều phản tác dụng. Phân tích cơ chế oc_update() cho thấy 'gate'
(cổng nhị phân stiff_ok AND vào bisection thể tích DUY NHẤT) sụp về lmid->0
khi vi phạm - Q được evaluate 1 LẦN tại x cũ và giữ NGUYÊN suốt 100 vòng
bisection nhị phân, nên `vol>volfrac and stiff_ok` luôn False bất kể lmid,
khiến TOÀN BỘ ràng buộc thể tích bị bỏ qua, mọi phần tử -dc>0 nhảy lên x+move
ĐỒNG LOẠT -> thể tích vọt lên -> vòng sau constraint đổi lại thuần volume ->
kéo mạnh xuống -> dao động (limit-cycle). 'dual' dùng multiplier RIÊNG cho
stiffness (nested bisection), không bao giờ bỏ hoàn toàn ràng buộc volume.

Bắt cặp CÙNG (volfrac, void_size_frac) giữa 2 config (giống PILOT_REPORT.md)
để so sánh công bằng, dùng ĐÚNG dải hẹp lịch sử đã verify cho hexagonal
(NARROW_VOLFRAC_RANGE=0.50-0.58, NARROW_VOID_SIZE_FRAC_RANGE=0.28-0.38, xem
generate_production_batch.py) và ĐÚNG tiêu chí "yield sạch" của script đó
(is_connected AND not osc_unstable AND v12<0 AND not degenerate).

Cách chạy:
    python3 analysis/scripts/pilot_oc_dual_multiplier.py --n 100
    python3 analysis/scripts/pilot_oc_dual_multiplier.py --n 400 --n-workers 10
"""
import os
import sys
import csv
import argparse
import time
import math
from multiprocessing import Pool, cpu_count

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "scripts"))

FIXED_PARAMETERS = dict(penal=3.0, rmin=1.5, move=0.2)
NARROW_VOLFRAC_RANGE = (0.50, 0.58)
NARROW_VOID_SIZE_FRAC_RANGE = (0.28, 0.38)

MANIFEST_FIELDS = [
    "config", "sample_id", "pair_idx", "volfrac", "void_size_frac",
    "v12", "v21", "converged", "n_iters", "hit_cap",
    "is_connected", "n_components", "degenerate", "volfrac_achieved",
    "osc_score", "osc_unstable", "is_clean", "elapsed_s",
]


def _worker(task):
    """Top-level function (cần cho multiprocessing pickle) - đúng convention
    bare-import của project (xem tests/conftest.py::_isolate_pipeline_bare_imports)."""
    pair_idx, config, volfrac, void_size_frac, run_dir = task

    from simp.runner import run_simp
    from manufacturability import check_connectivity
    from audit_label_stability import osc_score, OSC_THRESHOLD

    sample_id = f"{config}_{pair_idx:05d}"
    output_dir = os.path.join(run_dir, "simp_runs", sample_id)
    params = dict(
        nelx=50, nely=50, ft=2, E0=199.0, Emin=1e-9, nu=0.3,
        max_iter=150, tol_change=0.01, tol_obj=0.05, window_size=20,
        objective="auxetic", rotation_deg=0.0, beta=1.0,
        save_every=999, scale_factor=1, verbose=False,
        seed="hexagonal", volfrac=volfrac, void_size_frac=void_size_frac,
        output_dir=output_dir, stiffness_mode=config,
        **FIXED_PARAMETERS,
    )

    t0 = time.time()
    try:
        result = run_simp(params)
    except Exception as e:
        return {"sample_id": sample_id, "config": config, "pair_idx": pair_idx, "error": str(e)}
    elapsed = time.time() - t0

    xPhys = result["xPhys"]
    xphys_bin = (xPhys > 0.5).astype(np.float32)
    conn = check_connectivity(xphys_bin, min_feature_px=2)
    osc = osc_score(output_dir)
    hit_cap = result["n_iters"] >= params["max_iter"]
    volfrac_achieved = float(np.mean(xPhys))
    degenerate = volfrac_achieved < 0.05 or volfrac_achieved > 0.95
    osc_unstable = osc >= OSC_THRESHOLD
    v12 = result["v12"]
    is_clean = (
        conn["is_connected"] and not osc_unstable
        and not math.isnan(v12) and v12 < 0 and not degenerate
    )

    return {
        "config": config, "sample_id": sample_id, "pair_idx": pair_idx,
        "volfrac": volfrac, "void_size_frac": void_size_frac,
        "v12": v12, "v21": result["v21"], "converged": result["converged"],
        "n_iters": result["n_iters"], "hit_cap": hit_cap,
        "is_connected": conn["is_connected"], "n_components": conn["n_components"],
        "degenerate": degenerate, "volfrac_achieved": volfrac_achieved,
        "osc_score": osc, "osc_unstable": osc_unstable, "is_clean": is_clean,
        "elapsed_s": elapsed,
    }


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Số mẫu MỖI config (gate, dual)")
    parser.add_argument("--n-workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--run-dir", type=str,
                         default=os.path.join(REPO_ROOT, "outputs", "pilot_oc_dual_multiplier"))
    parser.add_argument("--seed-rng", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.run_dir, "simp_runs"), exist_ok=True)
    manifest_path = os.path.join(args.run_dir, "manifest.csv")

    rng = np.random.RandomState(args.seed_rng)
    pairs = [
        (float(rng.uniform(*NARROW_VOLFRAC_RANGE)), float(rng.uniform(*NARROW_VOID_SIZE_FRAC_RANGE)))
        for _ in range(args.n)
    ]
    tasks = []
    for i, (vf, vs) in enumerate(pairs):
        tasks.append((i, "gate", vf, vs, args.run_dir))
        tasks.append((i, "dual", vf, vs, args.run_dir))

    write_header = not os.path.exists(manifest_path)
    t0 = time.time()
    rows = []
    n_done = 0
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        with Pool(processes=args.n_workers) as pool:
            for row in pool.imap_unordered(_worker, tasks, chunksize=2):
                n_done += 1
                if "error" in row:
                    print(f"[{n_done}/{len(tasks)}] LỖI {row['sample_id']}: {row['error']}", flush=True)
                    continue
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if n_done % 20 == 0 or n_done == len(tasks):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
                    print(f"[{n_done}/{len(tasks)}] {rate:.2f} mẫu/s | ETA {eta/60:.1f} phút", flush=True)

    print(f"\nXong: {n_done} mẫu, {time.time()-t0:.0f}s. Manifest: {manifest_path}")

    for config in ("gate", "dual"):
        sub = [r for r in rows if r["config"] == config]
        n = len(sub)
        k_clean = sum(1 for r in sub if r["is_clean"])
        k_degen = sum(1 for r in sub if r["degenerate"])
        k_osc = sum(1 for r in sub if r["osc_unstable"])
        k_discon = sum(1 for r in sub if not r["is_connected"])
        lo, hi = wilson_ci(k_clean, n)
        print(f"{config:5s}  n={n:4d}  sạch={k_clean:4d} ({100*k_clean/max(1,n):.1f}%)  "
              f"95%CI=[{100*lo:.1f}%,{100*hi:.1f}%]  suy_biến={100*k_degen/max(1,n):.1f}%  "
              f"dao_động={100*k_osc/max(1,n):.1f}%  rời_rạc={100*k_discon/max(1,n):.1f}%")


if __name__ == "__main__":
    main()
