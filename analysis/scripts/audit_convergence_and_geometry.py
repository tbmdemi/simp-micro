"""
Audit theo yêu cầu advisor (2026-07-24, xem EXPERIMENT_LOG.md):

1. "Coi các cấu trúc đạt 150 iteration là hội tụ" có đúng không?
   simp/core/convergence.py::ConvergenceChecker.should_stop() gán
   converged=True CẢ KHI loop>=max_iter (150), không chỉ khi tolerance
   (tol_change/tol_obj) thực sự thoả mãn - xem test_should_stop_max_iter.
   Script này join manifest.csv với n_iters/max_iter gốc trong từng
   batch_{i}_results.csv để tách 2 trường hợp: "hội tụ thật" (n_iters <
   max_iter, dừng sớm vì tolerance) vs "chạm trần" (n_iters == max_iter,
   converged=True chỉ vì hết ngân sách vòng lặp - KHÔNG chắc đã hội tụ).

2. Quét TOÀN BỘ ảnh cuối cùng (đúng ảnh Phase 3 dùng, sau BOX-resize 64x64
   khớp build_npz.py::load_and_resize) bằng
   pipeline/phase5_cvae/manufacturability.check_connectivity - phát hiện
   cấu trúc rời rạc (n_components>1) hoặc suy biến (toàn rỗng/toàn đặc).

Cross-tab 2 trục trên: "chạm trần" có tương quan với hình xấu/rời rạc không?

Chạy:
    python3 analysis/scripts/audit_convergence_and_geometry.py

Output: in bảng tổng hợp ra terminal + lưu
outputs/phase3/convergence_geometry_audit.json (chi tiết),
outputs/phase3/convergence_geometry_audit_flagged.csv (danh sách mẫu bị
flag: rời rạc hoặc suy biến), và outputs/phase3/manifest_quality.csv (bản
mở rộng của manifest.csv, thêm n_iters/max_iter/hit_cap/n_components/
is_connected/degenerate/solid_frac cho MỌI mẫu - notebooks/utils.py đọc lại
file này để sửa bug n_iter=150 cứng/converged_flag sai, xem
notebooks/01_data_loading_and_quality_control.ipynb).
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from manufacturability import check_connectivity  # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
MANIFEST_PATH = os.path.join(PHASE3_DIR, "manifest.csv")
RESOLUTION = 64  # khớp build_npz.py mặc định (dataset Phase 5 dùng)

# batch_id -> đường dẫn batch_{id}_results.csv (batch 11 nằm trong thư mục
# đặt tên khác vì là bản rebuild sau dq-fix, xem VERIFICATION_GUIDE.md).
BATCH_RESULTS_PATHS = {
    1: "outputs/multi_batch/batch_1/batch_1_results.csv",
    2: "outputs/multi_batch/batch_2/batch_2_results.csv",
    3: "outputs/multi_batch/batch_3/batch_3_results.csv",
    4: "outputs/multi_batch/batch_4/batch_4_results.csv",
    5: "outputs/multi_batch/batch_5/batch_5_results.csv",
    6: "outputs/multi_batch/batch_6/batch_6_results.csv",
    7: "outputs/multi_batch/batch_7/batch_7_results.csv",
    8: "outputs/multi_batch/batch_8/batch_8_results.csv",
    11: "outputs/multi_batch/batch_11_dqfix_full_rebuild/batch_11_results.csv",
}


def load_iters() -> pd.DataFrame:
    """Ghép n_iters/max_iter từ từng batch_{i}_results.csv, khoá nối
    (batch, seed, sample_id) - đúng khoá scan_dataset.py dùng để map sang
    manifest.csv (xem docstring scan_dataset.py về sample_id toàn cục)."""
    frames = []
    for batch_id, rel_path in BATCH_RESULTS_PATHS.items():
        df = pd.read_csv(os.path.join(REPO_ROOT, rel_path))
        df = df[["sample_id", "seed", "n_iters", "max_iter", "converged"]].copy()
        df["batch"] = batch_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest: {len(manifest)} mẫu")

    iters = load_iters()
    merged = manifest.merge(
        iters, on=["batch", "seed", "sample_id"], how="left",
        suffixes=("", "_raw"),
    )
    n_unmatched = merged["n_iters"].isna().sum()
    if n_unmatched:
        print(f"CẢNH BÁO: {n_unmatched} mẫu không khớp được n_iters "
              f"(bỏ qua phần join cho các mẫu này).")

    # ---- 1. Audit hội tụ ----
    merged["hit_cap"] = merged["n_iters"] >= merged["max_iter"]
    matched = merged.dropna(subset=["n_iters"])
    n_matched = len(matched)
    n_hit_cap = int(matched["hit_cap"].sum())
    n_true_converged = n_matched - n_hit_cap

    print("\n=== 1. Audit hội tụ (max_iter=150 mặc định Phase2) ===")
    print(f"Khớp được n_iters: {n_matched}/{len(manifest)}")
    print(f"  Hội tụ THẬT (n_iters < max_iter, dừng sớm vì tolerance): "
          f"{n_true_converged} ({n_true_converged/n_matched*100:.1f}%)")
    print(f"  Chạm trần (n_iters == max_iter, converged=True KHÔNG đảm bảo "
          f"tolerance đã thoả mãn): {n_hit_cap} ({n_hit_cap/n_matched*100:.1f}%)")
    iters_desc = matched["n_iters"].describe()
    print(f"  Phân phối n_iters: mean={iters_desc['mean']:.1f} "
          f"median={matched['n_iters'].median():.0f} "
          f"min={iters_desc['min']:.0f} max={iters_desc['max']:.0f}")

    print("\n  Theo seed (% chạm trần):")
    per_seed_cap = matched.groupby("seed")["hit_cap"].mean().sort_values(ascending=False)
    for seed, frac in per_seed_cap.items():
        print(f"    {seed:22s} {frac*100:5.1f}%")

    # ---- 2. Quét hình học toàn bộ dataset ----
    print(f"\n=== 2. Quét connectivity/hình học ({len(manifest)} ảnh, "
          f"resize {RESOLUTION}x{RESOLUTION} khớp build_npz.py) ===")
    n_components_list = []
    is_connected_list = []
    degenerate_list = []  # toàn rỗng hoặc toàn đặc (suy biến, không phải TOPO hợp lệ)
    solid_frac_list = []
    missing_img = 0

    for i, row in enumerate(manifest.itertuples()):
        img_path = os.path.join(REPO_ROOT, row.image_path) if isinstance(row.image_path, str) else None
        if img_path is None or not os.path.exists(img_path):
            missing_img += 1
            n_components_list.append(np.nan)
            is_connected_list.append(np.nan)
            degenerate_list.append(True)
            solid_frac_list.append(np.nan)
            continue
        im = Image.open(img_path).convert("L").resize((RESOLUTION, RESOLUTION), Image.BOX)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        solid_frac = float((arr > 0.5).mean())
        conn = check_connectivity(arr, min_feature_px=2)
        n_components_list.append(conn["n_components"])
        is_connected_list.append(conn["is_connected"])
        degenerate_list.append(solid_frac <= 0.001 or solid_frac >= 0.999)
        solid_frac_list.append(solid_frac)
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(manifest)}")

    manifest["n_components"] = n_components_list
    manifest["is_connected"] = is_connected_list
    manifest["degenerate"] = degenerate_list
    manifest["solid_frac"] = solid_frac_list

    n_disconnected = int((manifest["is_connected"] == False).sum())  # noqa: E712
    n_degenerate = int(manifest["degenerate"].sum())
    print(f"\nẢnh thiếu file: {missing_img}")
    print(f"Rời rạc (n_components > 1): {n_disconnected} "
          f"({n_disconnected/len(manifest)*100:.2f}%)")
    print(f"Suy biến (toàn rỗng/toàn đặc, solid_frac<=0.1% hoặc >=99.9%): "
          f"{n_degenerate} ({n_degenerate/len(manifest)*100:.2f}%)")

    print("\n  Rời rạc theo seed:")
    disc_by_seed = manifest.groupby("seed")["is_connected"].apply(
        lambda s: (s == False).mean()  # noqa: E712
    ).sort_values(ascending=False)
    for seed, frac in disc_by_seed.items():
        print(f"    {seed:22s} {frac*100:5.1f}%")

    # ---- 3. Cross-tab: chạm trần có tương quan với hình xấu không? ----
    merged_full = manifest.merge(
        matched[["batch", "seed", "sample_id", "hit_cap", "n_iters"]],
        on=["batch", "seed", "sample_id"], how="left",
    )
    ct = pd.crosstab(
        merged_full["hit_cap"].fillna("unmatched"),
        merged_full["is_connected"],
        dropna=False,
    )
    print("\n=== 3. Cross-tab: chạm trần (hit_cap) vs is_connected ===")
    print(ct)

    disc_rate_hit_cap = merged_full[merged_full["hit_cap"] == True]["is_connected"]  # noqa: E712
    disc_rate_converged_early = merged_full[merged_full["hit_cap"] == False]["is_connected"]  # noqa: E712
    if len(disc_rate_hit_cap) and len(disc_rate_converged_early):
        r1 = (disc_rate_hit_cap == False).mean()  # noqa: E712
        r2 = (disc_rate_converged_early == False).mean()  # noqa: E712
        print(f"\nTỷ lệ rời rạc trong nhóm CHẠM TRẦN: {r1*100:.2f}%")
        print(f"Tỷ lệ rời rạc trong nhóm HỘI TỤ SỚM:  {r2*100:.2f}%")

    # ---- Lưu output ----
    report = {
        "n_total": len(manifest),
        "convergence_audit": {
            "n_matched": n_matched,
            "n_true_converged_early": n_true_converged,
            "n_hit_cap": n_hit_cap,
            "frac_hit_cap": n_hit_cap / n_matched if n_matched else None,
            "n_iters_mean": float(iters_desc["mean"]),
            "n_iters_median": float(matched["n_iters"].median()),
            "per_seed_hit_cap_frac": per_seed_cap.to_dict(),
        },
        "geometry_audit": {
            "n_missing_img": missing_img,
            "n_disconnected": n_disconnected,
            "frac_disconnected": n_disconnected / len(manifest),
            "n_degenerate": n_degenerate,
            "frac_degenerate": n_degenerate / len(manifest),
            "per_seed_disconnected_frac": disc_by_seed.to_dict(),
        },
    }
    out_json = os.path.join(PHASE3_DIR, "convergence_geometry_audit.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo: {out_json}")

    flagged = manifest[(manifest["is_connected"] == False) | (manifest["degenerate"])]  # noqa: E712
    out_csv = os.path.join(PHASE3_DIR, "convergence_geometry_audit_flagged.csv")
    flagged.to_csv(out_csv, index=False)
    print(f"Đã lưu {len(flagged)} mẫu bị flag (rời rạc/suy biến): {out_csv}")

    # ---- manifest_quality.csv: manifest.csv gốc + n_iters/hit_cap thật +
    # kết quả quét hình học, để notebooks đọc lại thay vì hardcode/tính sai
    # (xem notebooks/utils.py::load_manifest_samples() trước khi sửa: gán
    # cứng n_iter=150 cho MỌI mẫu và converged_flag = copy thẳng cột
    # 'converged' vốn không phân biệt hội tụ thật vs chạm trần - xem mục 1).
    quality = manifest.merge(
        matched[["batch", "seed", "sample_id", "n_iters", "max_iter", "hit_cap"]],
        on=["batch", "seed", "sample_id"], how="left",
    )
    out_quality = os.path.join(PHASE3_DIR, "manifest_quality.csv")
    quality.to_csv(out_quality, index=False)
    print(f"Đã lưu manifest mở rộng ({len(quality)} mẫu, +n_iters/hit_cap/"
          f"n_components/is_connected/degenerate/solid_frac): {out_quality}")


if __name__ == "__main__":
    main()
