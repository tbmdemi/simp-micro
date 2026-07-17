"""
Phase 3 - Script 4/4: Lọc outlier, chia train/val/test, augment, xuất dataset cuối.

Các bước:
  1. Lọc mẫu suy biến: volfrac_achieved < 0.05 (topology sụp về rỗng) hoặc
     > 0.95 (đặc hoàn toàn) - không mang thông tin hình học hữu ích.
  2. Chia train/val/test = 70/15/15, STRATIFY theo seed (đảm bảo mỗi seed
     geometry xuất hiện tỉ lệ đều ở cả 3 tập, tránh val/test chỉ toàn 1-2
     seed nào đó).
  3. Augment CHỈ tập train bằng đối xứng vật lý (script 3), x6 kích thước.
  4. Lưu 3 file .npz riêng: train (augmented), val, test (nguyên bản,
     KHÔNG augment - để đánh giá phản ánh đúng phân phối thật).

Output:
  outputs/phase3/train.npz  (~70% x 6 augment)
  outputs/phase3/val.npz    (~15%, không augment)
  outputs/phase3/test.npz   (~15%, không augment)
  outputs/phase3/split_report.json  (thống kê để kiểm tra)
"""
import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(__file__))
from augment_symmetry import augment_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--vf-min", type=float, default=0.05,
                         help="Ngưỡng dưới volfrac_achieved để loại mẫu suy biến")
    parser.add_argument("--vf-max", type=float, default=0.95,
                         help="Ngưỡng trên volfrac_achieved để loại mẫu suy biến")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_path = os.path.join(PHASE3_DIR, f"dataset_{args.resolution}.npz")
    data = np.load(src_path, allow_pickle=True)

    n_total = len(data["images"])
    vf = data["volfrac_achieved"]
    converged = data["converged"]

    keep_mask = converged & (vf >= args.vf_min) & (vf <= args.vf_max)
    n_dropped = n_total - keep_mask.sum()
    print(f"Tổng mẫu: {n_total}, loại bỏ (suy biến/không converge): {n_dropped} "
          f"({100 * n_dropped / n_total:.2f}%)")

    idx_all = np.where(keep_mask)[0]
    seed_names_all = data["seed_names"][idx_all]

    # Chia stratify theo seed: train vs (val+test) trước, rồi val vs test
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=(args.val_frac + args.test_frac),
        stratify=seed_names_all, random_state=args.seed,
    )
    rel_test_frac = args.test_frac / (args.val_frac + args.test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=rel_test_frac,
        stratify=data["seed_names"][idx_temp], random_state=args.seed,
    )

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    def subset(idx):
        return {
            "images": data["images"][idx],
            "v12": data["v12"][idx],
            "v21": data["v21"][idx],
            "volfrac_achieved": data["volfrac_achieved"][idx],
            "seed_names": data["seed_names"][idx],
            "seed_onehot": data["seed_onehot"][idx],
            "params": data["params"][idx],
            "batch": data["batch"][idx],
        }

    train_raw = subset(idx_train)
    val_data = subset(idx_val)
    test_data = subset(idx_test)

    # Augment CHỈ train
    print("\nĐang augment tập train bằng đối xứng vật lý (x6)...")
    extra = {
        "seed_onehot": train_raw["seed_onehot"],
        "params": train_raw["params"],
        "seed_names": train_raw["seed_names"],
        "batch": train_raw["batch"],
        "volfrac_achieved": train_raw["volfrac_achieved"],
    }
    train_aug = augment_dataset(
        train_raw["images"], train_raw["v12"], train_raw["v21"], extra
    )

    def save(path, d):
        np.savez_compressed(path, **d, seed_classes=data["seed_classes"],
                             param_names=data["param_names"])
        print(f"  Lưu {path}: images={d['images'].shape}, "
              f"{os.path.getsize(path) / 1e6:.1f} MB")

    save(os.path.join(PHASE3_DIR, "train.npz"), train_aug)
    save(os.path.join(PHASE3_DIR, "val.npz"), val_data)
    save(os.path.join(PHASE3_DIR, "test.npz"), test_data)

    # Báo cáo kiểm tra
    report = {
        "n_total_raw": int(n_total),
        "n_dropped_degenerate": int(n_dropped),
        "n_train_before_augment": int(len(idx_train)),
        "n_train_after_augment": int(len(train_aug["images"])),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "seed_distribution_train": {
            s: int((train_raw["seed_names"] == s).sum())
            for s in data["seed_classes"]
        },
        "seed_distribution_val": {
            s: int((val_data["seed_names"] == s).sum())
            for s in data["seed_classes"]
        },
        "seed_distribution_test": {
            s: int((test_data["seed_names"] == s).sum())
            for s in data["seed_classes"]
        },
        "v12_range_train": [float(train_aug["v12"].min()), float(train_aug["v12"].max())],
        "v12_range_val": [float(val_data["v12"].min()), float(val_data["v12"].max())],
        "v12_range_test": [float(test_data["v12"].min()), float(test_data["v12"].max())],
    }
    report_path = os.path.join(PHASE3_DIR, "split_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo: {report_path}")


if __name__ == "__main__":
    main()