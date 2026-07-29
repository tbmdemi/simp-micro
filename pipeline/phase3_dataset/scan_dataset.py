"""
Phase 3 - Script 1/4: Quét toàn bộ multi-batch DOE results, tạo manifest.

Với mỗi mẫu (sample) trong 8 batch, script này:
  1. Xác định file ảnh density field cuối cùng (iteration lớn nhất) trong
     thư mục outputs/multi_batch/batch_{i}/{seed}/sample_{local_idx:04d}/
  2. Đọc volume_fraction cuối cùng đạt được từ iteration_data.csv (dùng
     làm target thứ 3 tạm thời thay cho f1, chưa có trong pipeline).
  3. Ghi ra manifest.csv chứa: batch, seed, sample_id, local_idx,
     image_path, v12, v21, volfrac_achieved, converged, + 5 tham số thiết kế.

Mapping sample_id -> local_idx: trong mỗi batch, sample_id được đánh số
liên tục toàn batch nhưng thư mục sample_XXXX được đánh số lại từ 0 theo
từng seed (đã verify bằng groupby cumcount khớp với thứ tự thư mục).

Output: outputs/phase3/manifest.csv
"""
import os
import glob
import re
import json
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MULTI_BATCH_DIR = os.path.join(REPO_ROOT, "outputs", "multi_batch")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


def discover_batches() -> dict[int, str]:
    """Tự động tìm tất cả thư mục batch_* trong outputs/multi_batch/.

    BUG FIX (2026-07-29): trước đây hardcode N_BATCHES=8 và giả định tên thư
    mục đúng bằng "batch_{id}" - bỏ sót batch_9_seedweighted_validation,
    batch_10_dqfix_rebuild, batch_11_dqfix_full_rebuild (đặc biệt batch 11 là
    bản rebuild sửa lỗi dQ quan trọng nhất dự án, xem EXPERIMENT_LOG.md).
    Giờ quét động theo pattern "batch_<id>[_hậu tố tùy ý]/batch_<id>_results.csv"
    để tự bắt được batch mới trong tương lai mà không cần sửa hằng số. Xem
    AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục D1.

    Returns:
        Dict batch_id (int) -> đường dẫn thư mục batch tương ứng.
    """
    pattern = re.compile(r"^batch_(\d+)(?:_.*)?$")
    found: dict[int, str] = {}
    for entry in sorted(os.listdir(MULTI_BATCH_DIR)):
        full = os.path.join(MULTI_BATCH_DIR, entry)
        if not os.path.isdir(full):
            continue
        m = pattern.match(entry)
        if not m:
            continue
        batch_id = int(m.group(1))
        results_csv = os.path.join(full, f"batch_{batch_id}_results.csv")
        if not os.path.exists(results_csv):
            continue
        if batch_id in found:
            raise ValueError(
                f"Batch id {batch_id} trùng ở 2 thư mục: '{found[batch_id]}' và "
                f"'{full}' - cần xử lý thủ công (giữ lại bản nào?) trước khi quét."
            )
        found[batch_id] = full
    return found


def find_final_iteration_png(sample_dir: str) -> str | None:
    """Tìm file iteration_XXXXX.png có số iteration lớn nhất trong thư mục mẫu."""
    pngs = glob.glob(os.path.join(sample_dir, "iteration_*.png"))
    if not pngs:
        return None
    def iter_num(p):
        m = re.search(r"iteration_(\d+)\.png", os.path.basename(p))
        return int(m.group(1)) if m else -1
    pngs.sort(key=iter_num)
    return pngs[-1]


def final_volfrac(sample_dir: str) -> float | None:
    """Đọc Volume_Fraction ở dòng cuối của iteration_data.csv."""
    csv_path = os.path.join(sample_dir, "iteration_data.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        return float(df["Volume_Fraction"].iloc[-1])
    except Exception:
        return None


def scan_batch(batch_id: int, batch_dir: str) -> pd.DataFrame:
    results_csv = os.path.join(batch_dir, f"batch_{batch_id}_results.csv")
    df = pd.read_csv(results_csv)

    # local_idx = thứ tự trong nhóm seed, khớp với tên thư mục sample_XXXX
    df["batch"] = batch_id

    rows = []
    for _, r in df.iterrows():
        # Tên thư mục sample_XXXX dùng sample_id TOÀN CỤC trong batch,
        # không phải chỉ số cục bộ theo từng seed.
        sample_dir = os.path.join(
            batch_dir, r["seed"], f"sample_{int(r['sample_id']):04d}"
        )
        img_path = find_final_iteration_png(sample_dir)
        vf_final = final_volfrac(sample_dir)
        rows.append({
            "batch": batch_id,
            "seed": r["seed"],
            "sample_id": r["sample_id"],
            "image_path": os.path.relpath(img_path, REPO_ROOT) if img_path else None,
            "v12": r["v12"],
            "v21": r["v21"],
            "volfrac_achieved": vf_final,
            "obj_value": r["obj_value"],
            "converged": r["converged"],
            "volfrac": r["volfrac"],
            "penal": r["penal"],
            "rmin": r["rmin"],
            "move": r["move"],
            "void_size_frac": r["void_size_frac"],
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    batches = discover_batches()
    if not batches:
        raise RuntimeError(f"Không tìm thấy batch nào trong {MULTI_BATCH_DIR}")
    print(f"Tìm thấy {len(batches)} batch: {sorted(batches.keys())}")
    all_batches = []
    for b, batch_dir in sorted(batches.items()):
        print(f"Đang quét batch {b} ({os.path.basename(batch_dir)})...")
        all_batches.append(scan_batch(b, batch_dir))
    manifest = pd.concat(all_batches, ignore_index=True)

    n_missing_img = manifest["image_path"].isna().sum()
    n_missing_vf = manifest["volfrac_achieved"].isna().sum()
    print(f"\nTổng số mẫu: {len(manifest)}")
    print(f"Thiếu ảnh density: {n_missing_img}")
    print(f"Thiếu volfrac_achieved: {n_missing_vf}")

    out_path = os.path.join(OUT_DIR, "manifest.csv")
    manifest.to_csv(out_path, index=False)
    print(f"\nĐã lưu manifest: {out_path}")


if __name__ == "__main__":
    main()