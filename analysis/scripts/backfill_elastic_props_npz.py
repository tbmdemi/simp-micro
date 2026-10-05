"""
Backfill toàn bộ tính chất "rẻ" suy từ tensor Q cho dataset A4
(outputs/phase3_a4/{train,val,test}.npz - dataset của checkpoint 8D, có ν0
riêng từng mẫu trong cột `nu`).

"Rẻ" = không cần solver vật lý mới: chỉ 1 lần FE-solve đàn hồi + đồng nhất
hóa (đúng bài toán pipeline đã có), rồi mọi đại lượng là hàm đại số của Q:
  - E_x, E_y, G_xy, B_eff        (compute_elastic_constants, chia E0)
  - Q11/E0, Q22/E0               (mô-đun ràng buộc ngang = proxy độ cứng
                                  chống ấn lõm theo trục; trong plane stress
                                  đẳng hướng Q11 = E/(1-nu^2), chính là đại
                                  lượng trong proxy Hertz H ~ [E/(1-nu^2)]^g)
  - c_qL/c_qT theo x, y          (compute_wave_speeds, giới hạn bước sóng dài)
  - rel_density                  (mean ảnh trên lưới FE)
Lưu CẢ Q thô (n,3,3) để notebook suy thêm đại lượng khác không cần chạy lại.

Cùng phương pháp đã kiểm chứng ở backfill_f1_f2_npz.py (EXPERIMENT_LOG
2026-08-05): FE THẲNG trên ảnh 64x64 trong npz (resize nearest về lưới 50x50),
penal THẬT từng mẫu (params[:,1]), không join qua manifest. Khác: dùng thêm
ν0 THẬT từng mẫu (`nu`) vì dataset A4 vary vật liệu nền. Sanity: v12/v21 tính
lại phải khớp giá trị lưu sẵn (in median/p99 |Δ|).

Cách chạy (conda env 'simp'):
    python3 analysis/scripts/backfill_elastic_props_npz.py --split test
    python3 analysis/scripts/backfill_elastic_props_npz.py --split val
    python3 analysis/scripts/backfill_elastic_props_npz.py --split train

Output: outputs/phase3_a4/{split}_props.npz - cùng thứ tự hàng với
{split}.npz gốc (ghép theo chỉ số hàng, không copy ảnh).
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

A4_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_a4")

# Thứ tự cột scalar trong output - cố định để notebook đọc ổn định.
SCALAR_KEYS = (
    "E_x",
    "E_y",
    "G_xy",
    "B_eff",
    "M_x",
    "M_y",
    "c_qL_x",
    "c_qT_x",
    "c_qL_y",
    "c_qT_y",
    "rel_density",
    "v12_recheck",
    "v21_recheck",
)

_IMAGES = None
_PENAL = None
_NU0 = None


def _init_worker(images, penal, nu0):
    """Chia sẻ mảng dữ liệu cho worker qua biến global (fork, copy-on-write)
    thay vì pickle ảnh theo từng task."""
    global _IMAGES, _PENAL, _NU0
    _IMAGES, _PENAL, _NU0 = images, penal, nu0


def _solve_one(i: int):
    """FE-solve mẫu thứ i và suy mọi tính chất từ Q.

    Args:
        i: chỉ số hàng trong npz.

    Returns:
        (i, Q (3,3), dict scalar theo SCALAR_KEYS, chuỗi lỗi hoặc "") - lỗi
        FE/Q suy biến trả Q=NaN thay vì làm chết cả pool.
    """
    from verify_fe import FE_PARAMS, evaluate_density_field, resize_to_fe_grid

    from simp.objectives.auxetic import (
        compute_elastic_constants,
        compute_wave_speeds,
    )

    fe_params = dict(FE_PARAMS)
    fe_params["penal"] = float(_PENAL[i])
    fe_params["nu"] = float(_NU0[i])
    E0 = fe_params["E0"]
    try:
        img_fe = resize_to_fe_grid(
            _IMAGES[i], fe_params["nely"], fe_params["nelx"]
        )
        v12, v21, Q = evaluate_density_field(img_fe, fe_params)
        rel_density = float(img_fe.mean())
        ec = compute_elastic_constants(Q)
        ws = compute_wave_speeds(Q, rel_density, E0)
        row = {
            "E_x": ec["E_x"] / E0,
            "E_y": ec["E_y"] / E0,
            "G_xy": ec["G_xy"] / E0,
            "B_eff": ec["B_eff"] / E0,
            "M_x": float(Q[0, 0]) / E0,
            "M_y": float(Q[1, 1]) / E0,
            **ws,
            "rel_density": rel_density,
            "v12_recheck": float(v12),
            "v21_recheck": float(v21),
        }
        return i, Q, row, ""
    except Exception as e:
        return i, np.full((3, 3), np.nan), None, str(e)


def run(split: str, workers: int, limit: int = None) -> None:
    """Backfill 1 split và ghi {split}_props.npz.

    Args:
        split: 'train' | 'val' | 'test'.
        workers: số tiến trình song song.
        limit: chỉ chạy n mẫu đầu (smoke test), None = toàn bộ.
    """
    data = np.load(os.path.join(A4_DIR, f"{split}.npz"), allow_pickle=True)
    images = data["images"]
    penal = data["params"][:, 1]
    nu0 = data["nu"]
    n = len(images) if limit is None else min(limit, len(images))
    print(f"[{split}] n={n}, {workers} worker...", flush=True)

    Q_all = np.full((n, 3, 3), np.nan)
    cols = {k: np.full(n, np.nan) for k in SCALAR_KEYS}
    errors = 0
    with mp.Pool(
        workers, initializer=_init_worker, initargs=(images, penal, nu0)
    ) as pool:
        for count, (i, Q, row, err) in enumerate(
            pool.imap_unordered(_solve_one, range(n), chunksize=32)
        ):
            Q_all[i] = Q
            if err:
                errors += 1
            else:
                for k in SCALAR_KEYS:
                    cols[k][i] = row[k]
            if (count + 1) % 5000 == 0:
                print(f"  [{split}] ... {count + 1}/{n}", flush=True)

    # Sanity: v12 tính lại phải khớp giá trị gốc (sai khác chỉ do resize
    # 64->50, xem LIMITATIONS/EXPERIMENT_LOG f1/f2 backfill).
    ok = ~np.isnan(cols["v12_recheck"])
    dv12 = np.abs(data["v12"][:n][ok] - cols["v12_recheck"][ok])
    print(
        f"[{split}] lỗi FE: {errors}/{n}. |v12_gốc - v12_recheck|: "
        f"median={np.median(dv12):.4f}, p99={np.percentile(dv12, 99):.4f}, "
        f"frac>0.1={np.mean(dv12 > 0.1):.3%}",
        flush=True,
    )

    out_path = os.path.join(A4_DIR, f"{split}_props.npz")
    np.savez_compressed(out_path, Q=Q_all, keys=np.array(SCALAR_KEYS), **cols)
    print(f"[{split}] Đã lưu: {out_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", required=True, choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 1)
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.split, args.workers, args.limit)


if __name__ == "__main__":
    main()
