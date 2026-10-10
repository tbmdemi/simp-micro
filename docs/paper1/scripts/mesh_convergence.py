"""Hội tụ lưới FE cho Bài #1 (review mục 9).

Giữ NGUYÊN hình học nhị phân của thiết kế ở lưới verify 50×50, chia mỗi
pixel thành k×k phần tử (k = 1, 2, 4 → lưới 50², 100², 200²) rồi giải lại
homogenization. Sai lệch ν giữa các k chỉ đến từ rời rạc hóa FE, không từ
hình học - trả lời câu hỏi reviewer "ν có phụ thuộc lưới 50×50 không?".

Mặc định: 30 thiết kế ngẫu nhiên (seed 0) của test.npz; tùy chọn thêm
thiết kế refined từ JSON có ảnh (P1.6c, `--images-json`).

Chạy từ gốc repo:
    python docs/paper1/scripts/mesh_convergence.py \
        --out outputs/phase5/plan_v3/p1_6x_mesh_convergence.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from verify_fe import (  # noqa: E402
    FE_PARAMS,
    evaluate_density_field,
    resize_to_fe_grid,
)

FACTORS = (1, 2, 4)


def nu_at_refinements(design50: np.ndarray) -> dict:
    """(ν12, ν21) của cùng 1 thiết kế nhị phân 50×50 ở mọi hệ số chia k.

    Args:
        design50: ảnh nhị phân (50, 50).

    Returns:
        dict {k: [ν12, ν21]} (None nếu FE lỗi ở mức đó).
    """
    out = {}
    for k in FACTORS:
        fine = np.kron(design50, np.ones((k, k)))
        params = dict(FE_PARAMS, nelx=50 * k, nely=50 * k)
        try:
            v12, v21, _ = evaluate_density_field(fine, params)
            out[k] = [float(v12), float(v21)]
        except Exception:
            out[k] = None
    return out


def main():
    """CLI: tính ν theo k cho các thiết kế, ghi JSON + in tóm tắt."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-designs", type=int, default=30)
    p.add_argument("--images-json", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    test = np.load(os.path.join(REPO_ROOT, "outputs", "phase3", "test.npz"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test["images"]), args.n_designs, replace=False)
    designs = [
        (
            "test",
            int(i),
            resize_to_fe_grid(
                (test["images"][i] > 0.5).astype(np.float32), 50, 50
            ),
        )
        for i in idx
    ]
    if args.images_json:
        with open(args.images_json) as f:
            pc = json.load(f)["per_condition"]
        for i, c in enumerate(pc[: args.n_designs]):
            img = np.asarray(c["refined_image"], dtype=np.float32)
            designs.append(("refined", i, resize_to_fe_grid(img, 50, 50)))

    with Pool(args.workers) as pool:
        results = pool.map(nu_at_refinements, [d[2] for d in designs])

    rows = []
    for (src, i, _), res in zip(designs, results):
        rows.append({"source": src, "index": i, "nu": res})
    # Sai lệch tuyệt đối so với lưới mịn nhất (k=4) - tham chiếu "hội tụ".
    summary = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [
            r
            for r in rows
            if r["source"] == src
            and all(r["nu"][k] is not None for k in FACTORS)
        ]
        ref = np.array([r["nu"][FACTORS[-1]] for r in sub])
        summary[src] = {"n": len(sub)}
        for k in FACTORS[:-1]:
            d = np.abs(np.array([r["nu"][k] for r in sub]) - ref)
            summary[src][f"k{k}_vs_k{FACTORS[-1]}"] = {
                "median_abs_dv12": float(np.median(d[:, 0])),
                "max_abs_dv12": float(d[:, 0].max()),
                "median_abs_dv21": float(np.median(d[:, 1])),
                "max_abs_dv21": float(d[:, 1].max()),
                "median_rel_dv12": float(
                    np.median(d[:, 0] / np.abs(ref[:, 0]))
                ),
            }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
