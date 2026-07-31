"""
Phase 8.3 - build_design_library.py
============================================================
Roadmap 8.3: thư viện thiết kế tuyển chọn - gọi
pipeline/phase5_cvae/best_of_n_eval.py::best_of_n() cho từng target
(stratified theo bin v12 từ test.npz), lưu ảnh PNG + manifest.

Lưu ý phạm vi: chọn "best" theo accuracy (FE thật) + manufacturability -
KHÔNG có composite score 3 trục (aesthetic chỉ tồn tại trên nhánh
feature/optional-multi-condition chưa merge). Xem EXPERIMENT_LOG.md mục
2026-07-31 cho kết quả đầy đủ.

Cách chạy:
    python3 analysis/scripts/build_design_library.py --n-targets 24 --n-samples 30

Output: outputs/phase5/design_library/design_XXX.png + manifest.json/.csv
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from dataset import CVAEDataset                      # noqa: E402
from best_of_n_eval import best_of_n                  # noqa: E402
from manufacturability import check_manufacturability  # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
LIBRARY_DIR = os.path.join(PHASE5_DIR, "design_library")


def select_stratified_conditions(n_targets: int, n_bins: int = 8, seed: int = 123):
    """Lấy (v12,v21) THẬT từ test.npz, stratified theo bin v12 - cùng cách
    làm với run_independent_fe_check.py::select_stratified_indices()."""
    test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))
    v12_all = np.array([test_ds[i][1][0].item() for i in range(len(test_ds))])

    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(v12_all.min(), v12_all.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(v12_all, bin_edges[1:-1]), 0, n_bins - 1)

    per_bin = max(1, n_targets // n_bins)
    selected = []
    for b in range(n_bins):
        candidates = np.where(bin_idx == b)[0]
        if len(candidates) == 0:
            continue
        take = min(per_bin, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    selected = sorted(selected[:n_targets])

    conditions = [test_ds[i][1].numpy() for i in selected]
    return conditions


def run(n_targets: int, n_samples: int, cvae_ckpt: str, out_dir: str, seed: int = 123):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    conditions = select_stratified_conditions(n_targets, seed=seed)
    print(f"Đã chọn {len(conditions)} target (stratified theo v12) từ test.npz.")
    print(f"Checkpoint: {cvae_ckpt} | thiết bị: {device}")

    manifest = []
    for k, cond in enumerate(conditions):
        png_path = os.path.join(out_dir, f"design_{k:03d}.png")
        result = best_of_n(
            cvae_ckpt, n_conditions=1, n_samples=n_samples, device=device,
            seed=seed, k_fe_verify=None, require_manufacturable=True,
            custom_condition=cond, save_best_png=png_path,
            apply_force_periodic=True,
        )
        if not result["per_condition"]:
            print(f"  [bỏ qua] target {k}: mọi FE solve lỗi, không có ứng viên.")
            continue
        row = result["per_condition"][0]

        # best_of_n() rơi về accuracy-only nếu KHÔNG có ứng viên manufacturable
        # nào trong pool (xem best_of_n_eval.py) - kiểm tra lại trực tiếp ảnh
        # đã lưu để biết chắc thiết kế cuối có sản xuất được hay là fallback.
        saved_img = np.asarray(Image.open(png_path), dtype=np.float32) / 255.0
        best_is_manufacturable = bool(
            check_manufacturability((saved_img > 0.5).astype(np.float32))["passes_all"]
        )

        entry = {
            "design_id": f"design_{k:03d}",
            "png_path": os.path.relpath(png_path, REPO_ROOT),
            "target_v12": row["target_v12"],
            "target_v21": float(cond[1]),
            "v12_achieved": row["v12_best"],
            "is_auxetic_target": row["is_auxetic_target"],
            "hit": bool(row["is_auxetic_target"] and row["v12_best"] < 0),
            "n_manufacturable": row["n_manufacturable"],
            "frac_manufacturable": row["frac_manufacturable"],
            "n_valid_samples": row["n_valid_samples"],
            "best_is_manufacturable": best_is_manufacturable,
        }
        manifest.append(entry)
        print(f"  {entry['design_id']}: target_v12={entry['target_v12']:+.4f} "
              f"achieved={entry['v12_achieved']:+.4f} hit={entry['hit']} "
              f"frac_manufacturable={entry['frac_manufacturable']:.2f} "
              f"best_manufacturable={best_is_manufacturable}")

    if not manifest:
        raise RuntimeError("Không có thiết kế nào được tạo thành công.")

    n_hits = sum(1 for e in manifest if e["hit"])
    n_auxetic = sum(1 for e in manifest if e["is_auxetic_target"])
    mean_frac_manuf = float(np.mean([e["frac_manufacturable"] for e in manifest]))
    n_confirmed_manufacturable = sum(1 for e in manifest if e["best_is_manufacturable"])

    summary = {
        "cvae_ckpt": cvae_ckpt,
        "n_targets_requested": n_targets,
        "n_designs_in_library": len(manifest),
        "n_samples_per_target": n_samples,
        "hit_rate": n_hits / n_auxetic if n_auxetic else float("nan"),
        "mean_frac_manufacturable": mean_frac_manuf,
        "n_confirmed_manufacturable_designs": n_confirmed_manufacturable,
        "n_accuracy_only_fallback": len(manifest) - n_confirmed_manufacturable,
        "designs": manifest,
    }

    manifest_json = os.path.join(out_dir, "manifest.json")
    with open(manifest_json, "w") as f:
        json.dump(summary, f, indent=2)

    manifest_csv = os.path.join(out_dir, "manifest.csv")
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\n=== Thư viện thiết kế: {len(manifest)} mẫu ===")
    print(f"  hit_rate               = {summary['hit_rate']:.3f}")
    print(f"  mean_frac_manufacturable = {summary['mean_frac_manufacturable']:.3f}")
    print(f"  Xác nhận manufacturable = {n_confirmed_manufacturable}/{len(manifest)} "
          f"({summary['n_accuracy_only_fallback']} fallback accuracy-only)")
    print(f"  Manifest: {manifest_json}")
    print(f"  Manifest CSV: {manifest_csv}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-targets", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--cvae-ckpt", type=str,
                         default=os.path.join(PHASE5_DIR, "cvae_realphysics.pt"))
    parser.add_argument("--out-dir", type=str, default=LIBRARY_DIR)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run(args.n_targets, args.n_samples, args.cvae_ckpt, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
