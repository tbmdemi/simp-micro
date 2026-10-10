"""C5 (bỏ `force_periodic`): chạy lại phép đối chiếu composite score ↔ tầng
Pareto của notebook 08 cũ (`08_composite_score_pareto_validation.ipynb`,
đã xóa 2026-10-08, còn trong git history) trên pipeline không fp. Script
này là bản hiện hành của phép đối chiếu đó.

Giữ nguyên mọi thứ khác của notebook 08 (24 target seed 123, N=30, trọng số
0,6/0,3/0,1, xếp tầng Pareto kiểu NSGA-II trên 3 điểm số thô, Spearman giữa
composite và −tầng). Chỉ đổi: `apply_force_periodic=False` và
`periodicity_tol=1.0` - dung sai 100% làm cờ `periodic_ok` luôn = 1, tức
định nghĩa chế tạo được = liên thông + nét tối thiểu (quyết định C5). Hằng
số cộng thêm vào manuf_score không đổi thứ hạng trong pool, nên khác biệt so
với số cũ chỉ đến từ việc bỏ fp và bỏ kiểm tra cạnh.

Chạy từ gốc repo (GPU nếu có):
    python docs/paper1/scripts/c5_pareto_nofp.py
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, REPO_ROOT)

from best_of_n_eval import best_of_n  # noqa: E402
from dataset import CVAEDataset  # noqa: E402

from analysis.pareto.frontier import is_pareto_efficient  # noqa: E402

CKPT = os.path.join(REPO_ROOT, "outputs", "phase5", "cvae_v2_finetuned.pt")
OUT = os.path.join(
    REPO_ROOT, "outputs", "phase5", "plan_v3", "p1_9_pareto_nofp.json"
)


def non_domination_ranks(costs: np.ndarray) -> np.ndarray:
    """Tầng Pareto kiểu NSGA-II (1 = front), cực đại hóa mọi cột - chép đúng
    hàm của notebook 08 để số so sánh được."""
    ranks = np.zeros(costs.shape[0], dtype=int)
    remaining = np.arange(costs.shape[0])
    rank = 1
    while len(remaining) > 0:
        mask = is_pareto_efficient(costs[remaining], maximize=True)
        ranks[remaining[mask]] = rank
        remaining = remaining[~mask]
        rank += 1
    return ranks


def main():
    """Sinh 24×30 ứng viên không fp, tính Spearman composite ↔ −tầng."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ds = CVAEDataset(
        os.path.join(REPO_ROOT, "outputs", "phase3", "test.npz")
    )
    idxs = np.random.default_rng(123).choice(len(test_ds), 24, replace=False)
    rhos = []
    for i in idxs:
        r = best_of_n(
            CKPT,
            n_conditions=1,
            n_samples=30,
            device=device,
            seed=123,
            custom_condition=test_ds[i][1].numpy(),
            return_all_scores=True,
            apply_force_periodic=False,
            periodicity_tol=1.0,
        )
        sc = r["per_condition"][0]["all_scores"]
        costs = np.stack([sc["accuracy"], sc["manuf"], sc["aesthetic"]], 1)
        rho, _ = spearmanr(sc["composite"], -non_domination_ranks(costs))
        rhos.append(float(rho))
    rhos = np.array(rhos)
    out = {
        "n_conditions": len(rhos),
        "spearman_per_condition": rhos.tolist(),
        "mean_spearman": float(np.nanmean(rhos)),
        "median_spearman": float(np.nanmedian(rhos)),
        "min_spearman": float(np.nanmin(rhos)),
        "max_spearman": float(np.nanmax(rhos)),
        "frac_positive": float(np.mean(rhos > 0)),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "spearman_per_condition"},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
