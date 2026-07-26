"""
Sinh dataset raw mới bằng multiprocessing, dùng ĐÚNG production params
(FIXED_PARAMETERS/ACTIVE_PARAMETERS - pipeline/phase2_multi_batch/params.py),
KHÔNG bật move_min/penal_init (đã kiểm chứng không đáng tin cậy qua các
pilot trong phiên này). Đòn bẩy: chỉ dùng 3 seed năng suất cao đã kiểm
chứng trên manifest_quality.csv (bỏ điều kiện hit_cap - lý do: mislabeling
bug đã biết ở ConvergenceChecker khiến hit_cap không phản ánh chất lượng
label thật, xem osc_score gần 0 ở batch hourglass thử nghiệm dù hit_cap=True):
  hourglass         709/720 = 98.5% sạch (connected, osc<0.15, v12<0)
  hexagonal         676/720 = 93.9%
  reentrant_bowtie  556/720 = 77.2%

Ghi vào outputs/phase3_v2_raw/ (đĩa thật, ĐÃ có rule gitignore) - KHÔNG dùng
/tmp (tmpfs, mất dữ liệu qua reboot - đã xảy ra thật, mất ~10.700 mẫu/~3,5h
compute trong phiên rebuild v2, xem EXPERIMENT_LOG.md 2026-07-25). Chỉ npz
cuối cùng (sau khi build+split+augment, xem assemble_phase3_v2.py) mới ghi
vào outputs/phase3_v2/ rồi copy đè outputs/phase3/ khi chốt làm production.

Cách chạy:
    python3 analysis/scripts/generate_production_batch.py --n-raw 60  # test nhỏ
    python3 analysis/scripts/generate_production_batch.py --n-raw 9500 --resume
"""
import os
import sys
import csv
import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "scripts"))

FIXED_PARAMETERS = dict(penal=3.0, rmin=1.5, move=0.2)
SEED_WEIGHTS = {"hourglass": 0.70, "hexagonal": 0.15, "reentrant_bowtie": 0.15}
VOLFRAC_RANGE = (0.3, 0.7)
VOID_SIZE_FRAC_RANGE = (0.1, 0.4)

# BUG ĐÃ SỬA #2 (nghiêm trọng hơn, phát hiện sau batch beta=0.8): production
# pipeline thật (pipeline/phase2_multi_batch/runner.py::build_params_dict)
# KHÔNG BAO GIỜ set `beta` -> luôn dùng default beta=1.0 của run_simp(). Script
# này từng hardcode beta=0.8 (sai, không rõ nguồn gốc) suốt cả phiên - khiến
# reentrant_bowtie sụp về nghiệm suy biến gần như ngay lập tức (13 vòng,
# v12~0 tuyệt đối, 0% liên thông) vì beta thay đổi hoàn toàn cảnh quan mục
# tiêu. Đã xác minh: cùng volfrac=0.54/void_size_frac=0.33, beta=1.0 chạy đủ
# 150 vòng, v12=-0.063 (âm thật). ĐÃ SỬA beta=0.8 -> 1.0 bên dưới. TOÀN BỘ dữ
# liệu sinh trước khi sửa (2 batch, beta=0.8) không dùng được, phải sinh lại.

# BUG ĐÃ SỬA #1 (phát hiện sau khi chạy batch đầu, xem manifest.csv thực tế):
# hexagonal/reentrant_bowtie chỉ hội tụ tốt trong 1 dải volfrac/void_size_frac
# HẸP (~0.50-0.58 / ~0.28-0.38, khớp với manifest_quality.csv gốc - dataset cũ
# rõ ràng đã dùng dải hẹp này cho 2 seed này, không phải toàn bộ
# ACTIVE_PARAMETERS 0.3-0.7/0.1-0.4). Ngoài dải này, reentrant_bowtie sụp về
# nghiệm suy biến v12~0 gần như luôn luôn (đo được: 1.4% sạch ở dải rộng vs
# 77.2% ở dải hẹp lịch sử). hourglass KHÔNG bị ảnh hưởng (96.3% sạch ở dải
# rộng, thậm chí tốt hơn dải hẹp lịch sử 98.5% do n lớn hơn) nên vẫn dùng
# dải rộng để tối đa hóa đa dạng volfrac.
NARROW_VOLFRAC_RANGE = (0.50, 0.58)
NARROW_VOID_SIZE_FRAC_RANGE = (0.28, 0.38)
SEED_PARAM_RANGES = {
    "hourglass": (VOLFRAC_RANGE, VOID_SIZE_FRAC_RANGE),
    "hexagonal": (NARROW_VOLFRAC_RANGE, NARROW_VOID_SIZE_FRAC_RANGE),
    "reentrant_bowtie": (NARROW_VOLFRAC_RANGE, NARROW_VOID_SIZE_FRAC_RANGE),
}

MANIFEST_FIELDS = [
    "batch", "seed", "sample_id", "image_path", "v12", "v21", "obj_value",
    "converged", "volfrac", "penal", "rmin", "move", "void_size_frac",
    "n_components", "is_connected", "degenerate", "volfrac_achieved",
    "n_iters", "max_iter", "hit_cap", "osc_score", "osc_unstable",
]


def _worker(task):
    """Top-level function (cần cho multiprocessing pickle)."""
    idx, seed_name, volfrac, void_size_frac, run_dir = task

    # Import trong worker - mỗi process con cần tự import (tránh pickle lỗi
    # module state), theo đúng convention bare-import của project (xem
    # tests/conftest.py::_isolate_pipeline_bare_imports).
    from simp.runner import run_simp
    from manufacturability import check_connectivity
    from audit_label_stability import osc_score, OSC_THRESHOLD

    sample_id = f"{seed_name}_{idx:06d}"
    output_dir = os.path.join(run_dir, "simp_runs", sample_id)
    params = dict(
        nelx=50, nely=50, ft=2, E0=199.0, Emin=1e-9, nu=0.3,
        max_iter=150, tol_change=0.01, tol_obj=0.05, window_size=20,
        objective="auxetic", rotation_deg=0.0, beta=1.0,
        save_every=999, scale_factor=1, verbose=False,
        seed=seed_name, volfrac=volfrac, void_size_frac=void_size_frac,
        output_dir=output_dir,
        **FIXED_PARAMETERS,
    )

    try:
        result = run_simp(params)
    except Exception as e:
        return {"sample_id": sample_id, "error": str(e)}

    xPhys = result["xPhys"]
    xphys_bin = (xPhys > 0.5).astype(np.float32)
    conn = check_connectivity(xphys_bin, min_feature_px=2)
    osc = osc_score(output_dir)
    hit_cap = result["n_iters"] >= params["max_iter"]
    volfrac_achieved = float(np.mean(xPhys))
    degenerate = volfrac_achieved < 0.05 or volfrac_achieved > 0.95

    img_path = os.path.join(run_dir, "images", f"{sample_id}.png")
    Image.fromarray((np.clip(xPhys, 0, 1) * 255).astype(np.uint8)).save(img_path)

    return {
        "batch": 0, "seed": seed_name, "sample_id": sample_id,
        "image_path": img_path,
        "v12": result["v12"], "v21": result["v21"],
        "obj_value": result["objective"], "converged": result["converged"],
        "volfrac": volfrac, "penal": FIXED_PARAMETERS["penal"],
        "rmin": FIXED_PARAMETERS["rmin"], "move": FIXED_PARAMETERS["move"],
        "void_size_frac": void_size_frac,
        "n_components": conn["n_components"], "is_connected": conn["is_connected"],
        "degenerate": degenerate, "volfrac_achieved": volfrac_achieved,
        "n_iters": result["n_iters"], "max_iter": params["max_iter"],
        "hit_cap": hit_cap, "osc_score": osc, "osc_unstable": osc >= OSC_THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-raw", type=int, required=True)
    parser.add_argument("--n-workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--run-dir", type=str,
                         default=os.path.join(REPO_ROOT, "outputs", "phase3_v2_raw"),
                         help="Scratch bền vững (KHÔNG dùng /tmp - tmpfs mất qua reboot)")
    parser.add_argument("--seed-rng", type=int, default=42)
    parser.add_argument("--start-idx", type=int, default=0,
                         help="Offset sample_id (dùng khi resume/nối batch)")
    parser.add_argument("--only-seed", type=str, default=None,
                         help="Bỏ qua SEED_WEIGHTS, chỉ sinh 1 seed duy nhất (vd. hourglass)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.run_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "simp_runs"), exist_ok=True)
    manifest_path = os.path.join(args.run_dir, "manifest.csv")

    rng = np.random.RandomState(args.seed_rng)
    if args.only_seed:
        seeds, weights = [args.only_seed], [1.0]
    else:
        seeds = list(SEED_WEIGHTS.keys())
        weights = list(SEED_WEIGHTS.values())
    tasks = []
    for i in range(args.n_raw):
        idx = args.start_idx + i
        seed_name = rng.choice(seeds, p=weights)
        vf_range, vs_range = SEED_PARAM_RANGES[seed_name]
        volfrac = float(rng.uniform(*vf_range))
        void_size_frac = float(rng.uniform(*vs_range))
        tasks.append((idx, seed_name, volfrac, void_size_frac, args.run_dir))

    write_header = not os.path.exists(manifest_path)
    t0 = time.time()
    n_done, n_error, n_clean = 0, 0, 0
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        with Pool(processes=args.n_workers) as pool:
            for row in pool.imap_unordered(_worker, tasks, chunksize=4):
                n_done += 1
                if "error" in row:
                    n_error += 1
                    print(f"[{n_done}/{len(tasks)}] LỖI {row['sample_id']}: {row['error']}", flush=True)
                    continue
                writer.writerow(row)
                f.flush()
                is_clean = (
                    row["is_connected"] and not row["osc_unstable"]
                    and row["v12"] < 0 and not row["degenerate"]
                )
                n_clean += is_clean
                if n_done % 50 == 0 or n_done == len(tasks):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
                    print(f"[{n_done}/{len(tasks)}] sạch={n_clean} ({100*n_clean/n_done:.1f}%) "
                          f"lỗi={n_error} | {rate:.2f} mẫu/s | ETA {eta/60:.1f} phút", flush=True)

    print(f"\nXong: {n_done} mẫu, {n_clean} sạch ({100*n_clean/max(1,n_done):.1f}%), "
          f"{n_error} lỗi, {time.time()-t0:.0f}s")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
