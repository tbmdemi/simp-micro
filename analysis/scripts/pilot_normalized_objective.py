"""
Pilot A/B: objective_variant='q12' (mac dinh hien tai, mu=0.0) so voi
'normalized' (compute_auxetic_normalized_objective(), c = Q12/sqrt(Q11*Q22),
xem simp/objectives/auxetic.py:100-157 va simp/runner.py:263-267).

Boi canh (xem docs/LIMITATIONS.md muc 4, EXPERIMENT_LOG.md): mu penalty
tuyen tinh (c = Q12 - mu*(Q11+Q22)) da bi TAT (mu=0.0) vi sai sot khai niem
- runner.py:305 ghi ro "thuc nghiem cho thay mu > 0 khong cai thien (tham
chi Q12 duong hon)". Bien the 'normalized' la ung vien THAY THE da duoc
CAI DAT SAN nhung CHUA TUNG A/B test (docstring compute_auxetic_normalized_
objective(): "CHUA ap dung cho dataset hien co, can A/B test truoc"). Khong
can thiet ke cong thuc moi - chi can do xem no co thuc su day Q12 am hon
(auxetic manh hon) ma khong pha vo yield/stiffness hay khong.

Dung dung pattern voi pilot_mma.py (A/B tren 3 seed kho: hexagonal,
hourglass, reentrant_bowtie) - do:
  - yield (%sach = is_connected & not osc_unstable & v12<0 & not degenerate)
  - median/mean v12 dat duoc (auxetic manh yeu the nao)
  - Q11/Q22 trung binh (stiffness co bi hy sinh khong)

Cach chay (sau khi backfill f1/f2 xong, tranh tranh chap CPU/memory):
    python3 analysis/scripts/pilot_normalized_objective.py --seed hexagonal --n 100
    python3 analysis/scripts/pilot_normalized_objective.py --seed hourglass --n 100
    python3 analysis/scripts/pilot_normalized_objective.py --seed reentrant_bowtie --n 100
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

# Khop generate_production_batch.py / pilot_mma.py - dai tham so production
# that cua tung seed.
WIDE_VOLFRAC_RANGE = (0.3, 0.7)
WIDE_VOID_SIZE_FRAC_RANGE = (0.1, 0.4)
NARROW_VOLFRAC_RANGE = (0.50, 0.58)
NARROW_VOID_SIZE_FRAC_RANGE = (0.28, 0.38)
SEED_PARAM_RANGES = {
    "hourglass": (WIDE_VOLFRAC_RANGE, WIDE_VOID_SIZE_FRAC_RANGE),
    "hexagonal": (NARROW_VOLFRAC_RANGE, NARROW_VOID_SIZE_FRAC_RANGE),
    "reentrant_bowtie": (NARROW_VOLFRAC_RANGE, NARROW_VOID_SIZE_FRAC_RANGE),
}
FIXED_PARAMETERS = dict(penal=3.0, rmin=1.5, move=0.2)

MANIFEST_FIELDS = [
    "config", "sample_id", "pair_idx", "volfrac", "void_size_frac",
    "v12", "v21", "Q11", "Q22", "converged", "n_iters", "hit_cap",
    "is_connected", "n_components", "degenerate", "volfrac_achieved",
    "osc_score", "osc_unstable", "is_clean", "elapsed_s",
]


def _worker(task):
    pair_idx, config, seed_name, volfrac, void_size_frac, run_dir = task

    from manufacturability import check_connectivity
    from audit_label_stability import osc_score, OSC_THRESHOLD
    from simp.runner import run_simp

    sample_id = f"{config}_{pair_idx:05d}"
    output_dir = os.path.join(run_dir, "simp_runs", sample_id)

    params = dict(
        nelx=50, nely=50, ft=2, E0=199.0, Emin=1e-9, nu=0.3,
        max_iter=150, tol_change=0.01, tol_obj=0.05, window_size=20,
        objective="auxetic", rotation_deg=0.0, beta=1.0,
        save_every=999, scale_factor=1, verbose=False,
        seed=seed_name, volfrac=volfrac, void_size_frac=void_size_frac,
        output_dir=output_dir, stiffness_mode="gate",
        objective_variant=("normalized" if config == "normalized" else "q12"),
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
    Q = result["Q"]
    is_clean = (
        conn["is_connected"] and not osc_unstable
        and not math.isnan(v12) and v12 < 0 and not degenerate
    )

    return {
        "config": config, "sample_id": sample_id, "pair_idx": pair_idx,
        "volfrac": volfrac, "void_size_frac": void_size_frac,
        "v12": v12, "v21": result["v21"],
        "Q11": float(Q[0, 0]), "Q22": float(Q[1, 1]),
        "converged": result["converged"],
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
    parser.add_argument("--seed", type=str, required=True,
                         choices=list(SEED_PARAM_RANGES.keys()))
    parser.add_argument("--n", type=int, default=100, help="So mau MOI config (q12, normalized)")
    parser.add_argument("--n-workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Mac dinh outputs/pilot_normalized_<seed>/")
    parser.add_argument("--seed-rng", type=int, default=42)
    args = parser.parse_args()
    run_dir = args.run_dir or os.path.join(REPO_ROOT, "outputs", f"pilot_normalized_{args.seed}")

    os.makedirs(os.path.join(run_dir, "simp_runs"), exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.csv")

    vf_range, vs_range = SEED_PARAM_RANGES[args.seed]
    rng = np.random.RandomState(args.seed_rng)
    pairs = [
        (float(rng.uniform(*vf_range)), float(rng.uniform(*vs_range)))
        for _ in range(args.n)
    ]
    tasks = []
    for i, (vf, vs) in enumerate(pairs):
        tasks.append((i, "q12", args.seed, vf, vs, run_dir))
        tasks.append((i, "normalized", args.seed, vf, vs, run_dir))

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
                    print(f"[{n_done}/{len(tasks)}] LOI {row['sample_id']}: {row['error']}", flush=True)
                    continue
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if n_done % 20 == 0 or n_done == len(tasks):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
                    print(f"[{n_done}/{len(tasks)}] {rate:.2f} mau/s | ETA {eta/60:.1f} phut", flush=True)

    print(f"\nXong: {n_done} mau, {time.time()-t0:.0f}s. Manifest: {manifest_path}")

    for config in ("q12", "normalized"):
        sub = [r for r in rows if r["config"] == config]
        n = len(sub)
        k_clean = sum(1 for r in sub if r["is_clean"])
        v12_vals = [r["v12"] for r in sub if not math.isnan(r["v12"])]
        q11_vals = [r["Q11"] for r in sub]
        q22_vals = [r["Q22"] for r in sub]
        lo, hi = wilson_ci(k_clean, n)
        print(f"{config:12s}  n={n:4d}  sach={k_clean:4d} ({100*k_clean/max(1,n):.1f}%)  "
              f"95%CI=[{100*lo:.1f}%,{100*hi:.1f}%]  "
              f"v12_median={np.median(v12_vals):.4f}  "
              f"Q11_mean={np.mean(q11_vals):.2f}  Q22_mean={np.mean(q22_vals):.2f}")


if __name__ == "__main__":
    main()
