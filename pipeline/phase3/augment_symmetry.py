"""
Phase 3 - Script 3/4: Data augmentation bằng đối xứng hình học/vật lý.

Cơ sở vật lý: ô đơn vị (unit cell) tuần hoàn dưới tải trục 1 và trục 2.
  - Xoay 90/270 độ: trục 1 và trục 2 hoán đổi vai trò => v12_new = v21_old,
    v21_new = v12_old.
  - Lật ngang/dọc, xoay 180 độ: trục không hoán đổi => v12, v21 giữ nguyên
    (giả thiết tải đối xứng qua trục, đúng với homogenization chuẩn ở đây).

Chỉ áp dụng các phép hợp lệ vật lý (rotate90/180/270, flip_h, flip_v, tổ hợp)
- KHÔNG áp dụng phép biến đổi tuỳ ý (vd shear, cắt xén) vì sẽ phá vỡ tính
đúng đắn vật lý.

Áp dụng CHỈ trên tập train (tránh leakage giữa các phiên bản đối xứng của
cùng 1 mẫu vào cả train và val/test). Được gọi TỪ script 4 (finalize_dataset.py),
không chạy độc lập trên toàn bộ dataset.
"""
import numpy as np


def augment_sample(image: np.ndarray, v12: float, v21: float):
    """Sinh ra tối đa 4 biến thể đối xứng hợp lệ vật lý từ 1 mẫu.

    Trả về list các tuple (image_aug, v12_aug, v21_aug).
    Không dùng xoay 90/270 CÙNG với ảnh không vuông - ở đây ảnh luôn vuông
    (RES x RES) nên an toàn.
    """
    variants = []

    # Gốc
    variants.append((image, v12, v21))

    # Xoay 90 độ: hoán đổi trục 1<->2 => hoán đổi v12<->v21
    img_r90 = np.rot90(image, k=1)
    variants.append((img_r90, v21, v12))

    # Xoay 180 độ: trục không đổi vai trò => v12, v21 giữ nguyên
    img_r180 = np.rot90(image, k=2)
    variants.append((img_r180, v12, v21))

    # Xoay 270 độ: tương đương xoay -90, hoán đổi trục => hoán đổi v12<->v21
    img_r270 = np.rot90(image, k=3)
    variants.append((img_r270, v21, v12))

    # Lật ngang: trục không đổi vai trò => giữ nguyên v12, v21
    img_flip_h = np.fliplr(image)
    variants.append((img_flip_h, v12, v21))

    # Lật dọc: trục không đổi vai trò => giữ nguyên v12, v21
    img_flip_v = np.flipud(image)
    variants.append((img_flip_v, v12, v21))

    return variants


def augment_dataset(images: np.ndarray, v12: np.ndarray, v21: np.ndarray,
                     extra_arrays: dict, max_variants: int = 6):
    """Áp dụng augment_sample cho toàn bộ mảng, nhân bản extra_arrays tương ứng.

    extra_arrays: dict tên -> mảng (N, ...) các trường khác cần lặp lại
        theo số biến thể sinh ra cho mỗi mẫu (VD seed_onehot, params...).
    """
    n = len(images)
    out_images, out_v12, out_v21 = [], [], []
    out_extra = {k: [] for k in extra_arrays}

    for i in range(n):
        variants = augment_sample(images[i], float(v12[i]), float(v21[i]))[:max_variants]
        for img_v, v12_v, v21_v in variants:
            out_images.append(img_v)
            out_v12.append(v12_v)
            out_v21.append(v21_v)
            for k, arr in extra_arrays.items():
                out_extra[k].append(arr[i])

    result = {
        "images": np.stack(out_images).astype(np.float32),
        "v12": np.array(out_v12, dtype=np.float32),
        "v21": np.array(out_v21, dtype=np.float32),
    }
    for k in extra_arrays:
        result[k] = np.stack(out_extra[k])
    return result


if __name__ == "__main__":
    # Demo/self-test nhanh trên dữ liệu tổng hợp
    rng = np.random.default_rng(0)
    demo_img = rng.random((8, 8)).astype(np.float32)
    demo_v12, demo_v21 = -0.4, -0.6
    variants = augment_sample(demo_img, demo_v12, demo_v21)
    print(f"Sinh ra {len(variants)} biến thể từ 1 mẫu gốc:")
    for i, (img, v12, v21) in enumerate(variants):
        print(f"  variant {i}: shape={img.shape}, v12={v12:.3f}, v21={v21:.3f}")
    # Kiểm tra bất biến: xoay 90 hai lần phải bằng xoay 180
    assert np.allclose(np.rot90(demo_img, 2), np.rot90(np.rot90(demo_img, 1), 1))
    print("\nSelf-test OK: rot90 x2 == rot180")