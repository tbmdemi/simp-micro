"""Baseline retrieval trong phân phối trên IN100 (review mục 5).

Thư viện = outputs/phase3/train.npz (57 216 ảnh, đã augment). Với mỗi target
IN100: lấy k láng giềng gần nhất theo khoảng cách Euclid trên (ν12, ν21)
nhãn, verify TỪNG ảnh bằng đúng đường cVAE (force_periodic → ngưỡng 0,5 →
resize nearest 50² → FE) rồi:
  * retrieval-1   : láng giềng gần nhất (1 FE),
  * retrieval-k   : tốt nhất trong k theo sai số verified (k FE - cùng ngân
                    sách và cùng kiểu chọn FE-oracle với cVAE best-of-30).
Ghi cả nhãn thư viện để đo sai lệch nhãn ↔ verify (nhãn tính với penal
gốc của mẫu, verify dùng penal 3 cố định như cVAE).

Chạy từ gốc repo:
    python docs/paper1/scripts/retrieval_in100.py --k 30 \
        --out outputs/phase5/plan_v3/p1_6x_retrieval_in100.json
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

from benchmark_physics_guided_refinement import (  # noqa: E402
    _verify_v12_v21,
)
from verify_fe import FE_PARAMS  # noqa: E402

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
TARGETS_JSON = os.path.join(PLAN_V3, "p1_1e_b_v2_finetuned_in100_bo30.json")
_LIB = {}


def _init_worker(path: str, force_periodic: bool) -> None:
    """Mỗi worker nạp thư viện ảnh con 1 lần + cờ force_periodic."""
    _LIB["images"] = np.load(path)["images"]
    _LIB["fp"] = force_periodic


def _verify_index(i: int):
    """Verify 1 ảnh thư viện theo đúng đường cVAE best-of-30."""
    img = np.asarray(_LIB["images"][int(i)], dtype=np.float32)
    return _verify_v12_v21(img, FE_PARAMS, apply_force_periodic=_LIB["fp"])


def main():
    """CLI: retrieval-1 và retrieval-k verified cho IN100, ghi JSON."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--workers", type=int, default=10)
    # Mặc định bật để khớp đường verify cVAE best-of-30; tắt để đo riêng ảnh
    # hưởng của force_periodic (ảnh SIMP gốc vốn tuần hoàn qua PBC).
    p.add_argument("--no-force-periodic", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lib_path = os.path.join(REPO_ROOT, "outputs", "phase3", "train.npz")
    lib = np.load(lib_path)
    labels = np.stack([lib["v12"], lib["v21"]], axis=1).astype(np.float64)
    with open(TARGETS_JSON) as f:
        targets = np.array(
            [
                [c["target_v12"], c["target_v21"]]
                for c in json.load(f)["per_condition"]
            ]
        )

    nn_idx = np.argsort(
        ((labels[None, :, :] - targets[:, None, :]) ** 2).sum(-1), axis=1
    )[:, : args.k]
    flat = sorted(set(nn_idx.ravel().tolist()))
    tmp = os.path.join(PLAN_V3, "_retrieval_lib_tmp.npz")
    np.savez(tmp, images=lib["images"][flat])
    remap = {g: j for j, g in enumerate(flat)}
    with Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(tmp, not args.no_force_periodic),
    ) as pool:
        verified = dict(
            zip(flat, pool.map(_verify_index, [remap[g] for g in flat]))
        )
    os.remove(tmp)

    rows = []
    for t, idx in zip(targets, nn_idx):
        cands = []
        for g in idx:
            v = verified[int(g)]
            err = (
                float("inf")
                if v is None
                else abs(v[0] - t[0]) + abs(v[1] - t[1])
            )
            cands.append(
                {
                    "lib_index": int(g),
                    "label": labels[g].tolist(),
                    "verified": None if v is None else list(v),
                    "pair_err": err,
                }
            )
        best = min(cands, key=lambda c: c["pair_err"])
        rows.append(
            {"target": t.tolist(), "nearest": cands[0], "best_k": best}
        )

    def r2(p_, t_):
        return float(
            1 - ((p_ - t_) ** 2).sum() / ((t_ - t_.mean()) ** 2).sum()
        )

    summary = {}
    for key in ("nearest", "best_k"):
        P = np.array([r[key]["verified"] for r in rows])
        T = targets
        summary[key] = {
            "r2_v12": r2(P[:, 0], T[:, 0]),
            "r2_v21": r2(P[:, 1], T[:, 1]),
            "mae_v12": float(np.abs(P[:, 0] - T[:, 0]).mean()),
            "mae_v21": float(np.abs(P[:, 1] - T[:, 1]).mean()),
            "pair_mae": float(np.abs(P - T).sum(1).mean()),
            "fe_calls_per_target": 1 if key == "nearest" else args.k,
        }
    lab = np.array([r["nearest"]["label"] for r in rows])
    ver = np.array([r["nearest"]["verified"] for r in rows])
    summary["label_vs_verified_mae_v12"] = float(
        np.abs(lab[:, 0] - ver[:, 0]).mean()
    )
    summary["nearest_label_dist_median"] = float(
        np.median(np.linalg.norm(lab - targets, axis=1))
    )
    with open(args.out, "w") as f:
        json.dump(
            {
                "summary": summary,
                "config": {
                    "k": args.k,
                    "force_periodic": not args.no_force_periodic,
                },
                "rows": rows,
            },
            f,
            indent=1,
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
