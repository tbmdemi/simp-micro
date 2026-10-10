"""Ablation C5 (`LIMITATIONS.md` #34): có cần `force_periodic` không?

So 2 lần chạy y hệt (IN100, seed 123, best-of-30 + refine nhận thức nhị phân
hóa, cvae_v2_finetuned) chỉ khác cờ `--force-periodic`:
  * fp   : `outputs/phase5/plan_v3/p1_6c_in100_bo30_images.json`
  * no-fp: `outputs/phase5/plan_v3/p1_7_c5_in100_bo30_nofp.json`

Báo: độ chính xác verify (R², MAE ν12, sai số cặp) của baseline best-of-30
và bản guarded, CI ghép cặp no-fp vs fp; khả năng chế tạo tách 2 phần -
(a) liên thông + nét tối thiểu (độc lập với câu hỏi tuần hoàn) và (b) kiểm
tra cạnh đối diện khớp nhau (`check_periodicity`, chính là thứ bị nghi sai
khái niệm với lưới FE dựa trên phần tử + PBC).

Chạy từ gốc repo:
    python docs/paper1/scripts/c5_force_periodic.py
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from bootstrap_ci import bootstrap_paired_mae_reduction  # noqa: E402
from manufacturability import (  # noqa: E402
    check_connectivity,
    check_periodicity,
)

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
RUNS = {
    "fp": "p1_6c_in100_bo30_images.json",
    "nofp": "p1_7_c5_in100_bo30_nofp.json",
}


def final_arrays(per_condition: list) -> dict:
    """Mảng target, dự đoán baseline/guarded và cờ chế tạo của 1 lần chạy.

    Args:
        per_condition: per_condition của JSON refine có ảnh.

    Returns:
        dict: T (n,2), base (n,2), guard (n,2), conn_ok, period_ok (bool,
        trên ảnh nhị phân của thiết kế guarded).
    """
    T, B, G, conn, per = [], [], [], [], []
    for c in per_condition:
        T.append((c["target_v12"], c["target_v21"]))
        B.append((c["baseline_v12"], c["baseline_v21"]))
        key = "refined" if c["guarded_accept"] else "baseline"
        G.append((c[f"{key}_v12"], c[f"{key}_v21"]))
        img = np.asarray(c[f"{key}_image"])
        conn.append(check_connectivity(img)["manufacturable"])
        per.append(check_periodicity(img)["periodic_ok"])
    return {
        "T": np.array(T, float),
        "base": np.array(B, float),
        "guard": np.array(G, float),
        "conn_ok": np.array(conn),
        "period_ok": np.array(per),
    }


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    """R² = 1 − SS_res/SS_tot."""
    return float(
        1
        - ((pred - target) ** 2).sum() / ((target - target.mean()) ** 2).sum()
    )


def main():
    """Tính và in so sánh fp vs no-fp, ghi JSON tóm tắt."""
    data = {}
    for name, fname in RUNS.items():
        with open(os.path.join(PLAN_V3, fname)) as f:
            data[name] = final_arrays(json.load(f)["per_condition"])
    assert np.allclose(data["fp"]["T"], data["nofp"]["T"])
    T = data["fp"]["T"]

    out = {}
    for name, d in data.items():
        out[name] = {}
        for mode in ("base", "guard"):
            P = d[mode]
            out[name][mode] = {
                "r2_v12": r2(P[:, 0], T[:, 0]),
                "r2_v21": r2(P[:, 1], T[:, 1]),
                "mae_v12": float(np.abs(P[:, 0] - T[:, 0]).mean()),
                "pair_mae": float(np.abs(P - T).sum(1).mean()),
            }
        out[name]["frac_conn_minfeature"] = float(d["conn_ok"].mean())
        out[name]["frac_periodic_check"] = float(d["period_ok"].mean())
        out[name]["frac_passes_all"] = float(
            (d["conn_ok"] & d["period_ok"]).mean()
        )
    for mode in ("base", "guard"):
        e_fp = np.abs(data["fp"][mode] - T).sum(1)
        e_no = np.abs(data["nofp"][mode] - T).sum(1)
        # Dương = bỏ force_periodic giảm sai số cặp so với giữ nó.
        out[f"nofp_vs_fp_pair_{mode}"] = bootstrap_paired_mae_reduction(
            e_fp, e_no
        )
    path = os.path.join(PLAN_V3, "p1_7_c5_comparison.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
