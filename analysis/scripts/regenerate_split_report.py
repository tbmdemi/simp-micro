"""
Tính lại outputs/phase3/split_report.json TRỰC TIẾP từ train/val/test.npz
hiện có (không chạy lại pipeline sinh dữ liệu).

Lý do: split_report.json hiện tại (ghi bởi pipeline/phase3_dataset/finalize_dataset.py)
mô tả dataset CŨ trước "Rebuild v2" (5.258 mẫu raw, 11 seed phân bố gần đều),
trong khi outputs/phase3/{train,val,test}.npz thật đã được thay bằng dataset v2
(analysis/scripts/assemble_phase3_v2.py, 2026-07-25: train=57.216, seed lệch
nặng về hourglass/hexagonal/reentrant_bowtie) - xem
AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục D4.

QUAN TRỌNG: script này CHỈ đọc lại .npz cuối cùng - không có quyền truy cập vào
dữ liệu "trước lọc" (raw trước khi assemble_phase3_v2.py::is_clean() lọc bỏ),
nên KHÔNG thể tính lại n_total_raw/n_dropped_degenerate như bản cũ (những số
đó chỉ có trong log console lúc chạy assemble_phase3_v2.py, xem EXPERIMENT_LOG.md
"2026-07-25 - Rebuild dataset v2"). Các trường đó được để null thay vì bịa số.

Output: outputs/phase3/split_report.json (backup bản cũ trước khi ghi đè).
"""
import os
import json
import shutil

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


def _load(split):
    return np.load(os.path.join(PHASE3_DIR, f"{split}.npz"), allow_pickle=True)


def _seed_distribution(d):
    names = d["seed_names"]
    unique, counts = np.unique(names, return_counts=True)
    return {str(u): int(c) for u, c in zip(unique, counts)}


def _v12_range(d):
    return [float(d["v12"].min()), float(d["v12"].max())]


def main():
    train = _load("train")
    val = _load("val")
    test = _load("test")

    report = {
        "_note": (
            "Tính lại 2026-07-29 trực tiếp từ train/val/test.npz hiện có "
            "(dataset v2, assemble_phase3_v2.py) - KHÔNG phải chạy lại pipeline "
            "gốc. n_total_raw/n_dropped_degenerate của bản v2 không thể suy "
            "ngược từ .npz cuối cùng, xem EXPERIMENT_LOG.md mục 'Rebuild v2' "
            "để lấy số liệu đó. Xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục D4."
        ),
        "n_train_after_augment": int(len(train["images"])),
        "n_val": int(len(val["images"])),
        "n_test": int(len(test["images"])),
        "seed_distribution_train": _seed_distribution(train),
        "seed_distribution_val": _seed_distribution(val),
        "seed_distribution_test": _seed_distribution(test),
        "v12_range_train": _v12_range(train),
        "v12_range_val": _v12_range(val),
        "v12_range_test": _v12_range(test),
        "v12_all_negative": bool((train["v12"] < 0).all()
                                  and (val["v12"] < 0).all()
                                  and (test["v12"] < 0).all()),
    }

    out_path = os.path.join(PHASE3_DIR, "split_report.json")
    if os.path.exists(out_path):
        backup_path = os.path.join(PHASE3_DIR, "split_report_pre_v2_backup.json")
        if not os.path.exists(backup_path):
            shutil.copy2(out_path, backup_path)
            print(f"Đã backup bản cũ: {backup_path}")
        else:
            print(f"Backup đã tồn tại, không ghi đè: {backup_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu: {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
