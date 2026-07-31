"""
Vẽ 2 hình cho mục "Coverage toàn miền" của
outputs/phase5/reports/bao_cao_so_lieu_fix1_fix2.tex - dùng số liệu thật từ
pipeline/phase5_cvae/coverage_eval.py (checkpoint cvae_realphysics.pt, grid
8 target v12 trải đều [-0.81, 0.37], force_periodic BẬT/TẮT).

Bảng màu khớp NGUYÊN VĂN preamble của report (không dùng palette mặc định
của skill dataviz - đây là tài liệu ĐANG CÓ palette riêng, phải nhất quán):
serBlue #2a78d6 (trước/off), serOrange #eb6834 (sau/on, khớp cách report
highlight ô "sau" bằng serOrange!15), inkPrimary/Secondary/Muted, gridLine.
Cặp serBlue/serOrange đã validate qua scripts/validate_palette.js (skill
dataviz) - PASS mọi tiêu chí (CVD normal-vision floor ΔE=33.6).

Cách chạy:
    python3 analysis/scripts/plot_coverage_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "phase5", "reports", "figures")
RESULT_DIR = os.path.join(REPO_ROOT, "outputs", "phase5", "self_play")

SER_BLUE = "#2a78d6"
SER_ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_LINE = "#e1e0d9"

for _name in ("Noto Sans",):
    if any(_name.lower() in f.lower() for f in fm.findSystemFonts()):
        plt.rcParams["font.family"] = _name
        break

plt.rcParams.update({
    "axes.edgecolor": GRID_LINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "font.size": 11,
})


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_LINE)
    ax.spines["bottom"].set_color(GRID_LINE)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_frac_manufacturable(fp_on, fp_off, out_path):
    targets = [t["target_v12"] for t in fp_on["per_target"]]
    frac_on = [t["frac_manufacturable"] for t in fp_on["per_target"]]
    frac_off = [t["frac_manufacturable"] for t in fp_off["per_target"]]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    ax.axvline(0, color=GRID_LINE, linewidth=1.0, zorder=1)
    ax.plot(targets, frac_off, color=SER_BLUE, linewidth=2, marker="o",
            markersize=5, zorder=3, label="force_periodic TẮT")
    ax.plot(targets, frac_on, color=SER_ORANGE, linewidth=2, marker="o",
            markersize=5, zorder=3, label="force_periodic BẬT")

    ax.annotate("force_periodic BẬT", xy=(targets[-1], frac_on[-1]),
                xytext=(6, 4), textcoords="offset points",
                color=SER_ORANGE, fontsize=10, fontweight="bold")
    ax.annotate("force_periodic TẮT", xy=(targets[-1], frac_off[-1]),
                xytext=(6, -10), textcoords="offset points",
                color=SER_BLUE, fontsize=10, fontweight="bold")

    ax.set_xlabel("Target $\\nu_{12}$ (lưới 8 điểm, toàn miền train)")
    ax.set_ylabel("frac_manufacturable")
    ax.set_ylim(-0.03, 0.75)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_abs_error(fp_on, out_path):
    targets = [t["target_v12"] for t in fp_on["per_target"]]
    errors = [t["abs_error"] for t in fp_on["per_target"]]
    hits = [t["hit"] for t in fp_on["per_target"]]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    ax.axvline(0, color=GRID_LINE, linewidth=1.0, zorder=1)
    ax.plot(targets, errors, color=SER_ORANGE, linewidth=2, zorder=3)
    for x, y, hit in zip(targets, errors, hits):
        marker = "o" if hit is not False else "X"
        color = SER_ORANGE if hit is not False else "#c0392b"
        ax.scatter([x], [y], color=color, s=45, zorder=4, marker=marker)

    ax.set_xlabel("Target $\\nu_{12}$ (lưới 8 điểm, toàn miền train)")
    ax.set_ylabel("$|\\nu_{12}^{\\text{best}} - \\nu_{12}^{\\text{target}}|$")
    ax.set_ylim(0, max(errors) * 1.3)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    with open(os.path.join(RESULT_DIR, "coverage_realphysics_fp_on.json")) as f:
        fp_on = json.load(f)
    with open(os.path.join(RESULT_DIR, "coverage_realphysics_fp_off.json")) as f:
        fp_off = json.load(f)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_frac_manufacturable(fp_on, fp_off, os.path.join(FIG_DIR, "coverage_frac_manufacturable.pdf"))
    plot_abs_error(fp_on, os.path.join(FIG_DIR, "coverage_abs_error.pdf"))
    print("Đã lưu 2 hình vào", FIG_DIR)


if __name__ == "__main__":
    main()
