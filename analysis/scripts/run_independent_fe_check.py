"""
Phase 8.1 - run_independent_fe_check.py
============================================================
So sánh engine FE nội bộ (verify_fe.py::evaluate_density_field) với engine
độc lập scikit-fem (skfem_homogenization.py) trên mẫu thật (v12/v21/penal
đã biết) từ outputs/phase3/test.npz, không qua cVAE. Kết quả: xem
EXPERIMENT_LOG.md mục 2026-07-31.

Cách chạy:
    python3 analysis/scripts/run_independent_fe_check.py --n-samples 28

Output: outputs/phase5/reports/independent_fe_crosscheck.json
"""
import os
import sys
import json
import argparse

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, os.path.dirname(__file__))

from verify_fe import FE_PARAMS, resize_to_fe_grid, evaluate_density_field  # noqa: E402
import skfem_homogenization as skh                                          # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
REPORT_PATH = os.path.join(REPO_ROOT, "outputs", "phase5", "reports",
                            "independent_fe_crosscheck.json")


def select_stratified_indices(v12: np.ndarray, n_samples: int, n_bins: int = 8,
                               seed: int = 123):
    """Chia v12 thành n_bins khoảng đều, lấy ngẫu nhiên ~n_samples/n_bins
    mẫu/bin để mẫu so sánh trải đều property space."""
    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(v12.min(), v12.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(v12, bin_edges[1:-1]), 0, n_bins - 1)

    per_bin = max(1, n_samples // n_bins)
    selected = []
    for b in range(n_bins):
        candidates = np.where(bin_idx == b)[0]
        if len(candidates) == 0:
            continue
        take = min(per_bin, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())

    selected = selected[:n_samples]
    return sorted(selected)


def run(n_samples: int, out_path: str, seed: int = 123):
    test_path = os.path.join(PHASE3_DIR, "test.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Không tìm thấy {test_path} - script này cần outputs/phase3/test.npz "
            f"thật (mẫu SIMP đã sinh sẵn, có nhãn v12/v21/penal)."
        )
    raw = np.load(test_path, allow_pickle=True)
    images = raw["images"]
    v12_all = raw["v12"]
    v21_all = raw["v21"]
    param_names = list(raw["param_names"])
    penal_idx = param_names.index("penal")
    penal_all = raw["params"][:, penal_idx]

    indices = select_stratified_indices(v12_all, n_samples, seed=seed)
    print(f"Đã chọn {len(indices)} mẫu (stratified theo v12) từ "
          f"{len(v12_all)} mẫu test.npz.")

    rows = []
    for idx in indices:
        img64 = images[idx].astype(np.float32)
        img_fe = resize_to_fe_grid(img64, FE_PARAMS["nely"], FE_PARAMS["nelx"])
        fe_params_i = dict(FE_PARAMS, penal=float(penal_all[idx]))

        try:
            v12_existing, v21_existing, _Q_existing = evaluate_density_field(
                img_fe, fe_params_i,
            )
        except Exception as e:
            print(f"  [bỏ qua mẫu {idx}] engine hiện tại lỗi: {e}")
            continue

        try:
            v12_skfem, v21_skfem, _Q_skfem = skh.evaluate_density_field_skfem(
                img_fe, E0=fe_params_i["E0"], Emin=fe_params_i["Emin"],
                nu=fe_params_i["nu"], penal=fe_params_i["penal"],
            )
        except Exception as e:
            print(f"  [bỏ qua mẫu {idx}] engine scikit-fem lỗi: {e}")
            continue

        rows.append({
            "idx": int(idx),
            "penal": float(penal_all[idx]),
            "v12_saved_label": float(v12_all[idx]),
            "v12_existing": float(v12_existing),
            "v12_skfem": float(v12_skfem),
            "v21_existing": float(v21_existing),
            "v21_skfem": float(v21_skfem),
            "abs_diff_v12": float(abs(v12_existing - v12_skfem)),
            "abs_diff_v21": float(abs(v21_existing - v21_skfem)),
        })
        print(f"  mẫu {idx}: v12 hiện tại={v12_existing:+.4f} skfem={v12_skfem:+.4f} "
              f"(lệch={abs(v12_existing - v12_skfem):.4f}) | "
              f"v21 hiện tại={v21_existing:+.4f} skfem={v21_skfem:+.4f} "
              f"(lệch={abs(v21_existing - v21_skfem):.4f})")

    if not rows:
        raise RuntimeError("Không có mẫu nào chạy thành công ở cả 2 engine.")

    diffs_v12 = np.array([r["abs_diff_v12"] for r in rows])
    diffs_v21 = np.array([r["abs_diff_v21"] for r in rows])
    v12_existing_arr = np.array([r["v12_existing"] for r in rows])
    v12_skfem_arr = np.array([r["v12_skfem"] for r in rows])

    ss_res = ((v12_existing_arr - v12_skfem_arr) ** 2).sum()
    ss_tot = ((v12_existing_arr - v12_existing_arr.mean()) ** 2).sum()
    r2_v12 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    summary = {
        "n_samples": len(rows),
        "mean_abs_diff_v12": float(diffs_v12.mean()),
        "max_abs_diff_v12": float(diffs_v12.max()),
        "mean_abs_diff_v21": float(diffs_v21.mean()),
        "max_abs_diff_v21": float(diffs_v21.max()),
        "r2_v12_existing_vs_skfem": r2_v12,
        "fe_params": FE_PARAMS,
        "per_sample": rows,
    }

    print("\n=== Tổng kết đối chiếu FE độc lập (engine hiện tại vs scikit-fem) ===")
    print(f"  n_samples             = {summary['n_samples']}")
    print(f"  mean |Δv12|            = {summary['mean_abs_diff_v12']:.4f}")
    print(f"  max  |Δv12|            = {summary['max_abs_diff_v12']:.4f}")
    print(f"  mean |Δv21|            = {summary['mean_abs_diff_v21']:.4f}")
    print(f"  max  |Δv21|            = {summary['max_abs_diff_v21']:.4f}")
    print(f"  R²(v12, hiện tại vs skfem) = {summary['r2_v12_existing_vs_skfem']:.6f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nĐã lưu: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=28)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=REPORT_PATH)
    args = parser.parse_args()
    run(args.n_samples, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
