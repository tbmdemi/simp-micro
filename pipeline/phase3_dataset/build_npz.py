"""
Phase 3 - Script 2/4: Build dataset .npz từ manifest.csv

- Resize mỗi ảnh density field về RESOLUTION x RESOLUTION (mặc định 64x64),
  chuyển grayscale, normalize pixel về [0,1] (1 = vật liệu, 0 = lỗ rỗng).
  Dùng PIL Image.resize(..., Image.BOX) - phép lấy trung bình vùng, phù hợp
  khi downsample ảnh nhị phân/gần nhị phân hơn là nearest/bilinear.
- Target: v12, v21 (đã trong [-1, 1] về mặt vật lý, KHÔNG scale thêm để giữ
  ý nghĩa vật lý trực tiếp - hữu ích khi surrogate/cVAE cần suy luận ngược).
  volfrac_achieved thì scale MinMax [0,1] vì đã tự nhiên trong khoảng đó.
- Lưu kèm metadata (seed dạng one-hot + params thiết kế) để dùng cho
  conditional generation ở Phase 5.

Output: outputs/phase3/dataset_{RESOLUTION}.npz với các mảng:
    images          (N, RES, RES) float32, [0,1]
    v12, v21        (N,) float32
    volfrac_achieved(N,) float32
    seed_names      (N,) <U32  (string gốc, để tiện lọc/debug)
    seed_onehot     (N, n_seeds) float32
    params          (N, 5) float32  [volfrac, penal, rmin, move, void_size_frac]
    batch           (N,) int32
"""
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
MANIFEST_PATH = os.path.join(PHASE3_DIR, "manifest.csv")


def load_and_resize(image_path: str, resolution: int) -> np.ndarray:
    im = Image.open(image_path).convert("L")  # grayscale
    im = im.resize((resolution, resolution), Image.BOX)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64,
                         help="Kích thước ảnh density field vuông (mặc định 64)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Giới hạn số mẫu (debug nhanh, bỏ trống = toàn bộ)")
    args = parser.parse_args()
    res = args.resolution

    manifest = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        manifest = manifest.groupby("seed").head(
            max(1, args.limit // manifest["seed"].nunique())
        ).reset_index(drop=True)
    n = len(manifest)
    print(f"Đang xử lý {n} mẫu -> resolution {res}x{res}")

    images = np.zeros((n, res, res), dtype=np.float32)
    for i, row in enumerate(manifest.itertuples()):
        img_path = os.path.join(REPO_ROOT, row.image_path)
        images[i] = load_and_resize(img_path, res)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{n}")

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

    out_path = os.path.join(PHASE3_DIR, f"dataset_{res}.npz")
    np.savez_compressed(
        out_path,
        images=images,
        v12=v12,
        v21=v21,
        volfrac_achieved=volfrac_achieved,
        seed_names=seed_names,
        seed_onehot=seed_onehot,
        seed_classes=np.array(unique_seeds),
        params=params,
        param_names=np.array(["volfrac", "penal", "rmin", "move", "void_size_frac"]),
        batch=batch,
        converged=converged,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nĐã lưu: {out_path} ({size_mb:.1f} MB)")
    print(f"images shape: {images.shape}, dtype: {images.dtype}")
    print(f"v12 range: [{v12.min():.3f}, {v12.max():.3f}]")
    print(f"v21 range: [{v21.min():.3f}, {v21.max():.3f}]")
    print(f"volfrac_achieved range: [{volfrac_achieved.min():.3f}, {volfrac_achieved.max():.3f}]")
    print(f"seed classes ({len(unique_seeds)}): {unique_seeds}")


if __name__ == "__main__":
    main()