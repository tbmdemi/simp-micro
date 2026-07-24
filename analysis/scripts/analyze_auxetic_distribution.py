"""
Yêu cầu advisor (2026-07-24, xem EXPERIMENT_LOG.md): 1) xem phổ phân bố
auxetic (v12) từ -0.5 tới 0.3, tìm vùng tập trung nhiều nhất; 2) gán trọng
số cho các vùng muốn học nhiều hơn (vùng thưa mẫu) để dùng trong
WeightedRandomSampler của dataloader Phase 5 (xem dataset.py::compute_v12_bin_weights,
train.py --weighted-sampling).

Phương pháp trọng số: inverse-frequency theo bin histogram của v12 (mật độ
CÀNG THẤP thì trọng số CÀNG CAO, để dataloader lấy mẫu đều hơn qua toàn phổ
thay vì chỉ học tốt vùng mode) - power `alpha` điều chỉnh độ mạnh (1.0 =
nghịch đảo hoàn toàn, 0.5 = làm mượt bằng sqrt, 0.0 = tắt, giữ phân phối
gốc). Áp dụng trên train.npz (sau augment symmetry x6 - augment giữ nguyên
v12/v21 gốc nên không làm lệch histogram thêm).

Chạy:
    python3 analysis/scripts/analyze_auxetic_distribution.py

Output: in bảng phân bố ra terminal, lưu
outputs/figures/auxetic_properties/v12_distribution_weights.png và
outputs/phase3/v12_bin_weights.json (bin edges + weight, dataset.py đọc lại
file này để tính trọng số per-sample lúc train nếu --weighted-sampling).
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures", "auxetic_properties")

# Khoảng advisor yêu cầu xem ("-0.5 tới 0.3") - mở rộng nhẹ 2 đầu để không
# cắt mất phần đuôi phân phối thật (train v12 min~-1.95 do vài outlier hiếm,
# nhưng >99% mẫu nằm trong [-0.9, 0.65], xem in ra bên dưới).
BIN_EDGES = np.arange(-0.90, 0.70, 0.05)
ALPHA = 0.5  # inverse-frequency smoothing power (0.5 = sqrt, tránh trọng số quá cực đoan)


def compute_bin_weights(v12: np.ndarray, bin_edges: np.ndarray, alpha: float = 0.5):
    """weight(bin) = (1/count(bin))^alpha, chuẩn hoá mean=1. Mẫu ngoài
    [bin_edges[0], bin_edges[-1]] gán vào bin biên gần nhất (clip)."""
    counts, edges = np.histogram(v12, bins=bin_edges)
    counts_safe = np.maximum(counts, 1)  # tránh chia 0 cho bin rỗng
    raw_weight = counts_safe.astype(np.float64) ** (-alpha)
    bin_weight = raw_weight / raw_weight.mean()
    return bin_weight, counts, edges


def main():
    d = np.load(os.path.join(PHASE3_DIR, "train.npz"), allow_pickle=True)
    v12 = d["v12"].astype(np.float64)
    print(f"Train v12: n={len(v12)}, min={v12.min():.3f}, max={v12.max():.3f}, "
          f"mean={v12.mean():.3f}, median={np.median(v12):.3f}")
    frac_in_range = ((v12 >= -0.5) & (v12 <= 0.3)).mean()
    print(f"Tỷ lệ mẫu trong [-0.5, 0.3] (khoảng advisor hỏi): {frac_in_range*100:.1f}%")

    bin_weight, counts, edges = compute_bin_weights(v12, BIN_EDGES, ALPHA)

    print(f"\n{'bin':>18s} {'count':>7s} {'%':>6s} {'weight':>8s}")
    mode_idx = int(np.argmax(counts))
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        pct = counts[i] / len(v12) * 100
        marker = "  <- mode (tập trung nhiều nhất)" if i == mode_idx else ""
        print(f"[{lo:6.2f},{hi:6.2f}) {counts[i]:7d} {pct:5.1f}% {bin_weight[i]:8.3f}{marker}")

    print(f"\nVùng tập trung nhiều nhất: [{edges[mode_idx]:.2f}, {edges[mode_idx+1]:.2f}) "
          f"({counts[mode_idx]/len(v12)*100:.1f}% số mẫu)")
    sparse_mask = counts < counts.max() * 0.1
    sparse_ranges = [(edges[i], edges[i+1]) for i in range(len(counts)) if sparse_mask[i]]
    print(f"Vùng thưa (<10% mật độ đỉnh, được ưu tiên trọng số cao khi "
          f"weighted-sampling): {sparse_ranges}")

    # ---- Lưu weights để dataset.py/train.py dùng ----
    os.makedirs(PHASE3_DIR, exist_ok=True)
    weights_path = os.path.join(PHASE3_DIR, "v12_bin_weights.json")
    with open(weights_path, "w") as f:
        json.dump({
            "bin_edges": edges.tolist(),
            "bin_weight": bin_weight.tolist(),
            "bin_count": counts.tolist(),
            "alpha": ALPHA,
            "source": "train.npz v12",
        }, f, indent=2)
    print(f"\nĐã lưu bin weights: {weights_path}")

    # ---- Vẽ hình ----
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax1.bar(centers, counts, width=width * 0.9, color="#4c72b0", edgecolor="white")
    ax1.axvspan(-0.5, 0.3, color="orange", alpha=0.08, label="khoảng advisor hỏi [-0.5,0.3]")
    ax1.set_ylabel("Số mẫu (train, sau augment)")
    ax1.set_title("Phổ phân bố v12 (auxetic) - train.npz")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.bar(centers, bin_weight, width=width * 0.9, color="#c44e52", edgecolor="white")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel(f"Trọng số sampling (alpha={ALPHA})")
    ax2.set_xlabel("v12")
    ax2.set_title("Trọng số per-bin (nghịch đảo mật độ, mean=1)")

    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "v12_distribution_weights.png")
    fig.savefig(out_png, dpi=150)
    print(f"Đã lưu hình: {out_png}")


if __name__ == "__main__":
    main()
