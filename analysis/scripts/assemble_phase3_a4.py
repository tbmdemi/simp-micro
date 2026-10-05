"""
Lắp ráp dataset A4 (Giai đoạn A, docs/PROJECT_PLAN.md Nhóm 1): gộp pool sạch
CŨ (outputs/phase3/dataset_64.npz - chính là pool 13.624 mẫu đã dùng để build
outputs/phase3/{train,val,test}.npz hiện hành, tức baseline R²=0,974/0,964
"surrogate_v2.pt" - khớp CHÍNH XÁC split_report.json cũ, xem
EXPERIMENT_LOG.md 2026-08-05) với pool ν0 biến thiên MỚI (batch A4,
outputs/phase3_a4_nu_raw/, xem generate_production_batch.py --vary-nu).

Dùng dataset_64.npz làm nguồn "cũ" (KHÔNG dùng manifest_quality.csv như
assemble_phase3_v2.py) vì nó lớn hơn nhiều (13.624 so với 5.215 mẫu sạch từ
manifest_quality.csv) và LÀ pool THẬT đã tạo ra baseline cần so sánh - dùng
manifest_quality.csv sẽ làm train set A4 nhỏ hơn ~40% so với baseline một
cách không cần thiết, gây nhiễu lẫn giữa hiệu ứng "thêm ν0" và hiệu ứng
"ít dữ liệu hơn" khi so R².

seed_onehot của pool mới được build THEO ĐÚNG thứ tự seed_classes của pool cũ
(không tự sort độc lập) để đảm bảo 2 pool ghép cột one-hot đúng nghĩa.

Output: outputs/phase3_a4/{dataset_64,train,val,test}.npz + split_report.json
(gitignored, KHÔNG ghi đè outputs/phase3/ hiện tại).

Cách chạy:
    python3 analysis/scripts/assemble_phase3_a4.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase3_dataset"))
from augment_symmetry import augment_dataset  # noqa: E402
from finalize_dataset import _seed_only_stratify  # noqa: E402

OLD_DATASET_NPZ = os.path.join(REPO_ROOT, "outputs", "phase3", "dataset_64.npz")
RAW_MANIFEST = os.path.join(REPO_ROOT, "outputs", "phase3_a4_nu_raw", "manifest.csv")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_a4")
RESOLUTION = 64
OLD_NU_FALLBACK = 0.3  # dataset_64.npz được sinh với nu=0.3 cố định (trước Giai đoạn A)


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


def load_old_pool() -> dict:
    d = np.load(OLD_DATASET_NPZ, allow_pickle=True)
    n = len(d["images"])
    print(f"Old pool (dataset_64.npz, nu={OLD_NU_FALLBACK} cố định): {n} mẫu")
    return {
        "images": d["images"], "v12": d["v12"], "v21": d["v21"],
        "volfrac_achieved": d["volfrac_achieved"], "seed_names": d["seed_names"],
        "seed_onehot": d["seed_onehot"], "seed_classes": d["seed_classes"],
        "params": d["params"], "param_names": d["param_names"],
        "batch": d["batch"], "converged": d["converged"],
        "nu": np.full(n, OLD_NU_FALLBACK, dtype=np.float32),
    }


def build_new_pool(seed_classes: np.ndarray) -> dict:
    raw = pd.read_csv(RAW_MANIFEST)
    raw_clean = raw[is_clean(raw)].copy().reset_index(drop=True)
    raw_clean["image_path"] = raw_clean["image_path"].apply(resolve_path)
    missing = ~raw_clean["image_path"].apply(os.path.exists)
    if missing.any():
        print(f"CẢNH BÁO: {missing.sum()} ảnh không tồn tại, loại bỏ")
        raw_clean = raw_clean[~missing].reset_index(drop=True)

    unknown_seeds = set(raw_clean["seed"].unique()) - set(seed_classes.tolist())
    assert not unknown_seeds, (
        f"Batch A4 có seed lạ {unknown_seeds} không nằm trong seed_classes cũ "
        f"{seed_classes.tolist()} - one-hot sẽ lệch nếu ghép trực tiếp."
    )

    n = len(raw_clean)
    print(f"Raw pool mới (nu biến thiên): {len(raw)} -> sạch {n}")
    images = np.zeros((n, RESOLUTION, RESOLUTION), dtype=np.float32)
    for i, row in enumerate(raw_clean.itertuples()):
        images[i] = load_and_resize(row.image_path, RESOLUTION)
        if (i + 1) % 1000 == 0:
            print(f"  ... resize {i + 1}/{n}")

    seed_to_idx = {s: i for i, s in enumerate(seed_classes.tolist())}
    seed_onehot = np.zeros((n, len(seed_classes)), dtype=np.float32)
    for i, s in enumerate(raw_clean["seed"].to_numpy()):
        seed_onehot[i, seed_to_idx[s]] = 1.0

    return {
        "images": images,
        "v12": raw_clean["v12"].to_numpy(dtype=np.float32),
        "v21": raw_clean["v21"].to_numpy(dtype=np.float32),
        "volfrac_achieved": raw_clean["volfrac_achieved"].to_numpy(dtype=np.float32),
        "seed_names": raw_clean["seed"].to_numpy(),
        "seed_onehot": seed_onehot,
        "params": raw_clean[["volfrac", "penal", "rmin", "move", "void_size_frac"]].to_numpy(
            dtype=np.float32
        ),
        "batch": raw_clean["batch"].to_numpy(dtype=np.int32),
        "converged": raw_clean["converged"].to_numpy(dtype=bool),
        "nu": raw_clean["nu"].to_numpy(dtype=np.float32),
    }


def build_npz() -> str:
    old = load_old_pool()
    new = build_new_pool(old["seed_classes"])

    merged = {}
    for key in ("images", "v12", "v21", "volfrac_achieved", "seed_names",
                "seed_onehot", "params", "batch", "converged", "nu"):
        merged[key] = np.concatenate([old[key], new[key]], axis=0)
    n = len(merged["images"])
    print(f"Tổng dataset A4: {n} mẫu "
          f"(nu range: [{merged['nu'].min():.3f}, {merged['nu'].max():.3f}])")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"dataset_{RESOLUTION}.npz")
    np.savez_compressed(
        out_path, seed_classes=old["seed_classes"], param_names=old["param_names"],
        **merged,
    )
    print(f"Đã lưu: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
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
            "nu": data["nu"][idx],
        }

    train_raw = subset(idx_train)
    val_data = subset(idx_val)
    test_data = subset(idx_test)

    print("Đang augment tập train x6...")
    extra = {
        "seed_onehot": train_raw["seed_onehot"], "params": train_raw["params"],
        "seed_names": train_raw["seed_names"], "batch": train_raw["batch"],
        "volfrac_achieved": train_raw["volfrac_achieved"],
        "nu": train_raw["nu"],  # thuộc tính vật liệu - lặp lại y hệt volfrac_achieved
        # qua các biến thể đối xứng (không đổi bởi rotate/flip hình học).
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
        "nu_range": [float(data["nu"].min()), float(data["nu"].max())],
    }
    with open(os.path.join(OUT_DIR, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    dataset_path = build_npz()
    split_and_augment(dataset_path)


if __name__ == "__main__":
    main()
