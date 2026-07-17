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
N_BATCHES = 8


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


def scan_batch(batch_id: int) -> pd.DataFrame:
    batch_dir = os.path.join(MULTI_BATCH_DIR, f"batch_{batch_id}")
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
    all_batches = []
    for b in range(1, N_BATCHES + 1):
        print(f"Đang quét batch {b}...")
        all_batches.append(scan_batch(b))
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