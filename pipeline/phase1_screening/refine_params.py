"""
Sinh outputs/pipeline/phase1/refined_parameters.json từ _all_correlations.json:
với mỗi tham số SIMP (volfrac, penal, rmin, move, void_size_frac), quyết định
giữ ACTIVE (tiếp tục sample ở Phase 2) hay FIX ở giá trị nominal, dựa trên
p-value Spearman thu được từ 11 seeds ở Phase 1.

QUY TẮC: một tham số giữ ACTIVE nếu có ÍT NHẤT 1 seed đạt p < 0.10. Ngưỡng
lỏng (0.10 thay vì 0.05 chuẩn) để không bỏ sót tín hiệu borderline như `penal`
ở seed `reentrant_bowtie` (p≈0.06) — các seed có cơ chế đạt auxetic khác nhau
(vd bowtie dựa vào góc re-entrant hơn volfrac thuần túy), nên một tham số
"im lặng" ở 10 seed nhưng có tín hiệu ở 1 seed vẫn đáng giữ ACTIVE thay vì
cố định cứng.

Tham số FIXED được đặt ở TRUNG ĐIỂM range đã khảo sát trong pipeline/params.py
PARAM_SPACE — lựa chọn trung tính khi không có bằng chứng thống kê để chọn
giá trị cụ thể hơn (correlation không cho biết hướng tối ưu cục bộ chính xác).

Output schema (khớp pipeline/phase2_multi_batch/params.py::load_refined_parameters):
{
  "fixed_parameters": {"rmin": 1.75, ...},
  "active_parameters": {"volfrac": {"range": [0.45, 0.70]}, ...},
  "active_seeds": ["circle", "square", ..., "reentrant_bowtie"],
  "active_objectives": ["auxetic"],
  "_decision_log": {  # audit trail, không được multi_batch đọc, chỉ để tham khảo
      "volfrac": {"active": true, "min_pval": 1e-6, "seed": "square"},
      ...
  }
}

Usage:
    python -m pipeline.phase1_screening.refine_params \\
        --correlations outputs/pipeline/phase1/_all_correlations.json \\
        --output outputs/pipeline/phase1/refined_parameters.json
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

from pipeline.params import PARAM_SPACE

P_THRESHOLD = 0.10


def decide_active_params(correlations: Dict) -> Dict[str, Dict]:
    """Với mỗi param, tìm p-value nhỏ nhất trên toàn bộ seeds và quyết định.

    Args:
        correlations: nội dung đã parse của _all_correlations.json.

    Returns:
        Dict[param_name] -> {"active": bool, "min_pval": float,
                              "seed": str tên seed đạt min_pval}
    """
    param_names: List[str] = correlations["param_names"]
    decisions: Dict[str, Dict] = {
        p: {"active": False, "min_pval": float("inf"), "seed": None}
        for p in param_names
    }

    for cfg in correlations["configs"]:
        seed = cfg["seed"]
        for i, p in enumerate(param_names):
            pval = cfg["pval"][i]
            if pval < decisions[p]["min_pval"]:
                decisions[p]["min_pval"] = pval
                decisions[p]["seed"] = seed

    for p, d in decisions.items():
        d["active"] = d["min_pval"] < P_THRESHOLD
        if d["active"] and d["min_pval"] >= 0.05:
            d["note"] = f"borderline theo ngưỡng {P_THRESHOLD} (không đạt p<0.05 chuẩn)"

    return decisions


def build_refined_parameters(correlations: Dict) -> Dict:
    """Xây dựng nội dung refined_parameters.json đầy đủ."""
    decisions = decide_active_params(correlations)

    fixed_parameters: Dict[str, float] = {}
    active_parameters: Dict[str, Dict[str, List[float]]] = {}

    for param, decision in decisions.items():
        lo, hi = PARAM_SPACE[param]
        if decision["active"]:
            active_parameters[param] = {"range": [lo, hi]}
        else:
            fixed_parameters[param] = round((lo + hi) / 2, 4)

    active_seeds = sorted({cfg["seed"] for cfg in correlations["configs"]})
    active_objectives = sorted({cfg["objective"] for cfg in correlations["configs"]})

    return {
        "fixed_parameters": fixed_parameters,
        "active_parameters": active_parameters,
        "active_seeds": active_seeds,
        "active_objectives": active_objectives,
        "_decision_log": decisions,
        "_meta": {
            "p_threshold": P_THRESHOLD,
            "source": "pipeline/phase1_screening/refine_params.py",
        },
    }


def main(
    correlations_path: str = "outputs/pipeline/phase1/_all_correlations.json",
    output_path: str = "outputs/pipeline/phase1/refined_parameters.json",
) -> None:
    if not os.path.isfile(correlations_path):
        raise FileNotFoundError(
            f"Không tìm thấy {correlations_path}. Chạy Phase 1 aggregate "
            f"(--all) trước khi tạo refined_parameters.json."
        )

    with open(correlations_path) as f:
        correlations = json.load(f)

    refined = build_refined_parameters(correlations)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(refined, f, indent=2, ensure_ascii=False)

    # In tóm tắt quyết định ra console để review nhanh
    print(f"[INFO] Ghi {output_path}")
    print(f"[INFO] Ngưỡng p-value: {P_THRESHOLD}")
    print("\n  Tham số ACTIVE:")
    for p, r in refined["active_parameters"].items():
        d = refined["_decision_log"][p]
        note = f" ({d['note']})" if "note" in d else ""
        print(f"    {p:16s} range={r['range']}  min_p={d['min_pval']:.2e} @ {d['seed']}{note}")
    print("\n  Tham số FIXED:")
    for p, v in refined["fixed_parameters"].items():
        d = refined["_decision_log"][p]
        print(f"    {p:16s} = {v}  (min_p={d['min_pval']:.2e} @ {d['seed']}, không đạt p<{P_THRESHOLD})")
    print(f"\n  Seeds: {refined['active_seeds']}")
    print(f"  Objectives: {refined['active_objectives']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correlations",
        default="outputs/pipeline/phase1/_all_correlations.json",
        help="Đường dẫn _all_correlations.json (output của Phase 1 aggregate).",
    )
    parser.add_argument(
        "--output",
        default="outputs/pipeline/phase1/refined_parameters.json",
        help="Đường dẫn ghi refined_parameters.json.",
    )
    args = parser.parse_args()
    main(correlations_path=args.correlations, output_path=args.output)