"""
Lắp ráp dataset_v2: gộp manifest sạch từ batch mới (outputs/phase3_v2_raw/,
xem generate_production_batch.py) + manifest sạch cũ
(outputs/phase3/manifest_quality.csv), build npz 64x64, split 70/15/15
(stratify seed+v12), augment train x6.

Tiêu chí "sạch" THỐNG NHẤT cho cả 2 nguồn (xem phiên làm việc: hit_cap KHÔNG
dùng để lọc - mislabeling bug đã biết ở ConvergenceChecker):
    is_connected & osc_score < 0.15 & v12 < 0 & not degenerate

Output: outputs/phase3_v2/{dataset_64,train,val,test}.npz + split_report.json
(đã thêm rule gitignore, KHÔNG ghi đè outputs/phase3/ hiện tại).

Cách chạy:
    python3 analysis/scripts/assemble_phase3_v2.py
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase3_dataset"))
from augment_symmetry import augment_dataset  # noqa: E402
from finalize_dataset import _seed_only_stratify  # noqa: E402

RAW_MANIFEST = os.path.join(REPO_ROOT, "outputs", "phase3_v2_raw", "manifest.csv")
OLD_MANIFEST = os.path.join(REPO_ROOT, "outputs", "phase3", "manifest_quality.csv")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_v2")
RESOLUTION = 64


def is_clean(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_connected"].astype(bool)
        & (df["osc_score"] < 0.15)
        & (df["v12"] < 0)
        & (~df["degenerate"].astype(bool))
    )


def resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def load_and_resize(image_path: str, resolution: int) -> np.ndarray:
    im = Image.open(image_path).convert("L")
    im = im.resize((resolution, resolution), Image.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


def build_manifest() -> pd.DataFrame:
    raw = pd.read_csv(RAW_MANIFEST)
    old = pd.read_csv(OLD_MANIFEST)

    raw_clean = raw[is_clean(raw)].copy()
    old_clean = old[is_clean(old)].copy()
    print(f"Raw pool: {len(raw)} -> sạch {len(raw_clean)}")
    print(f"Old pool: {len(old)} -> sạch {len(old_clean)}")

    cols = ["image_path", "v12", "v21", "volfrac_achieved", "seed",
            "volfrac", "penal", "rmin", "move", "void_size_frac",
            "batch", "converged"]
    combined = pd.concat([raw_clean[cols], old_clean[cols]], ignore_index=True)
    combined["image_path"] = combined["image_path"].apply(resolve_path)
    missing = ~combined["image_path"].apply(os.path.exists)
    if missing.any():
        print(f"CẢNH BÁO: {missing.sum()} ảnh không tồn tại, loại bỏ")
        combined = combined[~missing].reset_index(drop=True)
    print(f"Tổng manifest sạch: {len(combined)}")
    return combined


def build_npz(manifest: pd.DataFrame) -> str:
    n = len(manifest)
    images = np.zeros((n, RESOLUTION, RESOLUTION), dtype=np.float32)
    for i, row in enumerate(manifest.itertuples()):
        images[i] = load_and_resize(row.image_path, RESOLUTION)
        if (i + 1) % 2000 == 0:
            print(f"  ... resize {i + 1}/{n}")

    v12 = manifest["v12"].to_numpy(dtype=np.float32)
    v21 = manifest["v21"].to_numpy(dtype=np.float32)
    volfrac_achieved = manifest["volfrac_achieved"].to_numpy(dtype=np.float32)
    seed_names = manifest["seed"].to_numpy()
    unique_seeds = sorted(manifest["seed"].unique())
    seed_to_idx = {s: i for i, s in enumerate(unique_seeds)}
    seed_onehot = np.zeros((n, len(unique_seeds)), dtype=np.float32)
    for i, s in enumerate(seed_names):
        seed_onehot[i, seed_to_idx[s]] = 1.0
    params = manifest[["volfrac", "penal", "rmin", "move", "void_size_frac"]].to_numpy(
        dtype=np.float32
    )
    batch = manifest["batch"].to_numpy(dtype=np.int32)
    converged = manifest["converged"].to_numpy(dtype=bool)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"dataset_{RESOLUTION}.npz")
    np.savez_compressed(
        out_path, images=images, v12=v12, v21=v21,
        volfrac_achieved=volfrac_achieved, seed_names=seed_names,
        seed_onehot=seed_onehot, seed_classes=np.array(unique_seeds),
        params=params, param_names=np.array(
            ["volfrac", "penal", "rmin", "move", "void_size_frac"]),
        batch=batch, converged=converged,
    )
    print(f"Đã lưu: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB), "
          f"seed classes: {unique_seeds}")
    return out_path


def split_and_augment(dataset_path: str, val_frac=0.15, test_frac=0.15, seed=42):
    data = np.load(dataset_path, allow_pickle=True)
    n_total = len(data["images"])
    idx_all = np.arange(n_total)
    seed_names_all = data["seed_names"][idx_all]
    v12_all = data["v12"][idx_all]

    strat_labels = _seed_only_stratify(seed_names_all, v12_all, n_bins=5)
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=(val_frac + test_frac),
        stratify=strat_labels, random_state=seed,
    )
    seed_names_temp = data["seed_names"][idx_temp]
    v12_temp = data["v12"][idx_temp]
    strat_labels_temp = _seed_only_stratify(seed_names_temp, v12_temp, n_bins=5)
    rel_test_frac = test_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=rel_test_frac,
        stratify=strat_labels_temp, random_state=seed,
    )
    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    def subset(idx):
        return {
            "images": data["images"][idx], "v12": data["v12"][idx],
            "v21": data["v21"][idx], "volfrac_achieved": data["volfrac_achieved"][idx],
            "seed_names": data["seed_names"][idx], "seed_onehot": data["seed_onehot"][idx],
            "params": data["params"][idx], "batch": data["batch"][idx],
        }

    train_raw = subset(idx_train)
    val_data = subset(idx_val)
    test_data = subset(idx_test)

    print("Đang augment tập train x6...")
    extra = {
        "seed_onehot": train_raw["seed_onehot"], "params": train_raw["params"],
        "seed_names": train_raw["seed_names"], "batch": train_raw["batch"],
        "volfrac_achieved": train_raw["volfrac_achieved"],
    }
    train_aug = augment_dataset(train_raw["images"], train_raw["v12"], train_raw["v21"], extra)

    def save(path, d):
        np.savez_compressed(path, **d, seed_classes=data["seed_classes"],
                             param_names=data["param_names"])

    save(os.path.join(OUT_DIR, "train.npz"), train_aug)
    save(os.path.join(OUT_DIR, "val.npz"), val_data)
    save(os.path.join(OUT_DIR, "test.npz"), test_data)

    report = {
        "n_total_clean_raw": int(n_total),
        "n_train_before_aug": int(len(idx_train)),
        "n_train_after_aug": int(len(train_aug["images"])),
        "n_val": int(len(idx_val)), "n_test": int(len(idx_test)),
        "seed_classes": data["seed_classes"].tolist(),
    }
    with open(os.path.join(OUT_DIR, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    manifest = build_manifest()
    manifest.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False) if os.path.isdir(OUT_DIR) else (
        os.makedirs(OUT_DIR, exist_ok=True) or manifest.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
    )
    dataset_path = build_npz(manifest)
    split_and_augment(dataset_path)


if __name__ == "__main__":
    main()
