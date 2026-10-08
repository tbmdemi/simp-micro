"""Đánh giá thiết kế CUỐI của từng phương pháp trên IN100 (review mục 8-9).

Với mỗi target IN100, lấy thiết kế cuối của:
  * cVAE best-of-30 + refine guarded (ảnh từ P1.6c `--save-images`),
  * SIMP best-of-4 seed, ngân sách `matched` và `full` (P1.6a `--save-images`),
rồi giải homogenization trên lưới verify 50² (k=1) VÀ lưới mịn 200² (k=4,
cùng hình học, chia mỗi pixel 4×4). Ghi ν12, ν21, E_x/E0, E_y/E0, thể tích.

Hai câu hỏi reviewer được trả lời:
  1. Độ chính xác có giữ được khi bỏ sai số rời rạc hóa 50² không (sai số
     50² vs 200² trung vị ~0,015 trên dữ liệu test - p1_6x_mesh_convergence)?
  2. Thiết kế auxetic có quá mềm (gần cơ cấu) không - E_x/E0 theo ν12.

Chạy từ gốc repo:
    python docs/paper1/scripts/final_design_eval.py \
        --out outputs/phase5/plan_v3/p1_6x_final_design_eval.json
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

from simp.objectives.auxetic import compute_elastic_constants  # noqa: E402

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
FACTORS = (1, 4)


def evaluate_design(design50: np.ndarray) -> dict:
    """ν và mô-đun (chia E0) của 1 thiết kế nhị phân 50×50 ở k = 1 và 4.

    Args:
        design50: ảnh nhị phân (50, 50).

    Returns:
        dict {"k1": {...}, "k4": {...}, "volume": float}; mỗi mức chứa
        v12, v21, E_x, E_y (đã chia E0) hoặc None nếu FE lỗi.
    """
    out = {"volume": float(design50.mean())}
    for k in FACTORS:
        fine = np.kron(design50, np.ones((k, k)))
        params = dict(FE_PARAMS, nelx=50 * k, nely=50 * k)
        try:
            _, _, Q = evaluate_density_field(fine, params)
            ec = compute_elastic_constants(Q)
            out[f"k{k}"] = {
                "v12": ec["nu_12"],
                "v21": ec["nu_21"],
                "E_x": ec["E_x"] / FE_PARAMS["E0"],
                "E_y": ec["E_y"] / FE_PARAMS["E0"],
            }
        except Exception:
            out[f"k{k}"] = None
    return out


def collect_designs(cvae_json: str, simp_json: str) -> list:
    """Gom (phương pháp, chỉ số target, target, ảnh 50×50) từ 2 file JSON.

    Ghép theo chỉ số condition (cùng thứ tự IN100, đã assert target khớp).
    """
    with open(cvae_json) as f:
        cv = json.load(f)["per_condition"]
    with open(simp_json) as f:
        sp = json.load(f)["per_condition"]
    designs = []
    for i, (c, s) in enumerate(zip(cv, sp)):
        assert abs(c["target_v12"] - s["target_v12"]) < 1e-9
        target = (c["target_v12"], c["target_v21"])
        key = "refined_image" if c["guarded_accept"] else "baseline_image"
        img = np.asarray(c[key], dtype=np.float32)
        designs.append(
            ("cVAE guarded", i, target, resize_to_fe_grid(img, 50, 50))
        )
        for bud in ("matched", "full"):
            img = np.asarray(s[bud]["best"]["image"], dtype=np.float32)
            designs.append((f"SIMP {bud} best-of-4", i, target, img))
    return designs


def summarize(rows: list) -> dict:
    """MAE/R² theo phương pháp ở mỗi lưới + trung vị E_x/E0 và thể tích."""

    def r2(p, t):
        return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())

    out = {}
    for m in sorted({r["method"] for r in rows}):
        sub = [
            r
            for r in rows
            if r["method"] == m
            and all(r["eval"][f"k{k}"] is not None for k in FACTORS)
        ]
        T = np.array([r["target"] for r in sub])
        out[m] = {"n": len(sub)}
        for k in FACTORS:
            P = np.array(
                [
                    [r["eval"][f"k{k}"]["v12"], r["eval"][f"k{k}"]["v21"]]
                    for r in sub
                ]
            )
            E = np.array([r["eval"][f"k{k}"]["E_x"] for r in sub])
            out[m][f"k{k}"] = {
                "r2_v12": r2(P[:, 0], T[:, 0]),
                "r2_v21": r2(P[:, 1], T[:, 1]),
                "mae_v12": float(np.abs(P[:, 0] - T[:, 0]).mean()),
                "mae_v21": float(np.abs(P[:, 1] - T[:, 1]).mean()),
                "median_E_x": float(np.median(E)),
                "frac_E_x_below_1e-3": float((E < 1e-3).mean()),
            }
        out[m]["median_volume"] = float(
            np.median([r["eval"]["volume"] for r in sub])
        )
    return out


def main():
    """CLI: đánh giá mọi thiết kế cuối, ghi JSON, in tóm tắt."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--cvae-json",
        default=os.path.join(PLAN_V3, "p1_6c_in100_bo30_images.json"),
    )
    p.add_argument(
        "--simp-json",
        default=os.path.join(PLAN_V3, "p1_6a_simp_baseline_in100.json"),
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    designs = collect_designs(args.cvae_json, args.simp_json)
    with Pool(args.workers) as pool:
        evals = pool.map(evaluate_design, [d[3] for d in designs])
    rows = [
        {"method": m, "index": i, "target": list(t), "eval": e}
        for (m, i, t, _), e in zip(designs, evals)
    ]
    summary = summarize(rows)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
