"""
plot_correlation_figures.py
─────────────────────────────
Xuất 5 PNG figures từ _all_correlations.json:
  1. heatmap_auxetic.png
  2. heatmap_first.png
  3. heatmap_second.png
  4. barplot_importance.png
  5. volcano_plot.png

Usage:
    python -m analysis.scripts.plot_correlation_figures
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ─── Configuration ───────────────────────────────────────────────────────────
CORRELATIONS_PATH = "outputs/pipeline/phase1/_all_correlations.json"
OUTPUT_DIR = "outputs/figures"

PARAM_LABELS = {
    "volfrac":        "Vol. Fraction",
    "penal":          "Penalty (p)",
    "rmin":           "Filter Radius",
    "move":           "Move Limit",
    "void_size_frac": "Void Size",
    "rotation_deg":   "Rotation (°)",
}
OBJ_LABELS = {
    "auxetic": "Auxetic Ratio",
    "first":   "1st Eigenfreq.",
    "second":  "2nd Eigenfreq.",
}
OBJ_COLORS = {"auxetic": "#e05f4e", "first": "#4e8fe0", "second": "#5dba6a"}
CMAP = "RdBu_r"
NORM = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


def load_data(path: str) -> dict:
    """Load the aggregated correlations JSON."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def draw_heatmap(data: dict, obj: str, filename: str, out_dir: Path) -> None:
    """Draw and save a Spearman correlation heatmap for one objective."""
    params = data["param_names"]
    configs = data["configs"]
    seeds = sorted(set(c["seed"] for c in configs if c["objective"] == obj))
    n_p = len(params)
    n_s = len(seeds)

    param_short = [PARAM_LABELS.get(p, p) for p in params]
    seed_short = [s.replace("_", "\n") for s in seeds]

    Z = np.array([[c["corr"][i] for c in configs
                    if c["seed"] == s and c["objective"] == obj][0]
                   for s in seeds for i in range(n_p)]).reshape(n_s, n_p)
    P = np.array([[c["pval"][i] for c in configs
                    if c["seed"] == s and c["objective"] == obj][0]
                   for s in seeds for i in range(n_p)]).reshape(n_s, n_p)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#f8f8f8")
    im = ax.imshow(Z, cmap=CMAP, norm=NORM, aspect="auto")

    for r in range(n_s):
        for c in range(n_p):
            sig = "*" if P[r, c] < 0.05 else ""
            val = f"{Z[r, c]:+.2f}{sig}"
            col = "white" if abs(Z[r, c]) > 0.6 else "#222"
            ax.text(c, r, val, ha="center", va="center",
                    fontsize=8, color=col,
                    fontweight="bold" if sig else "normal")

    ax.set_xticks(range(n_p))
    ax.set_xticklabels(param_short, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(seed_short, fontsize=8)
    ax.set_xlabel("Parameter", fontsize=10)
    ax.set_ylabel("Seed Topology", fontsize=10)

    ax.set_title(
        f"Spearman Correlation - {OBJ_LABELS[obj]}\n"
        f"(* p < 0.05 | {n_s} seeds × {n_p} params)",
        fontsize=12, fontweight="bold", color=OBJ_COLORS[obj], pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(str(out_dir / filename), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {filename}")


def draw_barplot(data: dict, out_dir: Path) -> None:
    """Draw grouped bar chart of mean absolute correlation per parameter."""
    params = data["param_names"]
    configs = data["configs"]
    objectives = sorted(set(c["objective"] for c in configs))
    seeds = sorted(set(c["seed"] for c in configs))
    n_p = len(params)

    param_short = [PARAM_LABELS.get(p, p) for p in params]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#f8f8f8")

    x = np.arange(n_p)
    bw = 0.26
    offs = [-bw, 0, bw]

    for k, obj in enumerate(objectives):
        means = np.array([
            np.mean([abs(c["corr"][i]) for c in configs
                     if c["objective"] == obj])
            for i in range(n_p)
        ])
        stds = np.array([
            np.std([abs(c["corr"][i]) for c in configs
                    if c["objective"] == obj])
            for i in range(n_p)
        ])
        ax.bar(x + offs[k], means, bw - 0.03,
               color=OBJ_COLORS[obj], alpha=0.88,
               label=OBJ_LABELS[obj],
               edgecolor="white", linewidth=0.6)
        ax.errorbar(x + offs[k], means, yerr=stds,
                    fmt="none", color="#333", capsize=4, linewidth=1.1)

    ax.axhline(0.3, color="#aaa", linestyle="--", linewidth=1.0)
    ax.axhline(0.6, color="#888", linestyle=":",  linewidth=1.0)
    ax.text(n_p - 0.05, 0.31, "moderate (0.3)", fontsize=7.5, color="#888",
            va="bottom", ha="right")
    ax.text(n_p - 0.05, 0.61, "strong (0.6)",   fontsize=7.5, color="#666",
            va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(param_short, fontsize=10)
    ax.set_ylabel("Mean |Spearman ρ| across seeds", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Parameter Importance - Mean Absolute Correlation by Objective\n"
        "(error bars = std across 10 seed topologies)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.75)
    ax.set_facecolor("#fefefe")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_dir / "barplot_importance.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ barplot_importance.png")


def draw_volcano(data: dict, out_dir: Path) -> None:
    """Draw volcano plot of correlation magnitude vs significance."""
    configs = data["configs"]
    params = data["param_names"]
    objectives = sorted(set(c["objective"] for c in configs))

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f8f8f8")

    for obj in objectives:
        rhos, pvals = [], []
        for cfg in configs:
            if cfg["objective"] != obj:
                continue
            for i in range(len(params)):
                rhos.append(cfg["corr"][i])
                pvals.append(cfg["pval"][i])
        rhos = np.array(rhos)
        pvals = np.array(pvals)
        ax.scatter(rhos, -np.log10(pvals + 1e-30),
                   c=OBJ_COLORS[obj], alpha=0.45, s=32,
                   edgecolors="none", label=OBJ_LABELS[obj])

    p05 = -np.log10(0.05)
    p01 = -np.log10(0.01)
    ax.axhline(p05, color="#e05f4e", linestyle="--", linewidth=1.3,
               label="p = 0.05")
    ax.axhline(p01, color="#a93226", linestyle=":",  linewidth=1.1,
               label="p = 0.01")
    ax.axvline(0, color="#ccc", linewidth=0.9)
    ax.axhspan(0, p05, alpha=0.07, color="gray")

    # Annotate notable points
    for obj in objectives:
        for cfg in configs:
            if cfg["objective"] != obj:
                continue
            for i, p in enumerate(params):
                r = cfg["corr"][i]
                pv = cfg["pval"][i]
                if abs(r) > 0.5 and pv < 0.05:
                    ax.annotate(
                        PARAM_LABELS.get(p, p),
                        (r, -np.log10(pv + 1e-30)),
                        fontsize=7, alpha=0.75, color="#333",
                        xytext=(4, 3), textcoords="offset points",
                    )

    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_ylabel("-log₁₀(p-value)", fontsize=10)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        "Volcano Plot - Correlation Magnitude vs Statistical Significance\n"
        "(annotated: |ρ| > 0.5 and p < 0.05)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.75, ncol=2)
    ax.set_facecolor("#fefefe")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(out_dir / "volcano_plot.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ volcano_plot.png")


def main() -> None:
    """Load aggregated data, generate all 5 PNG figures."""
    print("Loading correlation data…")
    data = load_data(CORRELATIONS_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    objectives = sorted(set(c["objective"] for c in data["configs"]))

    print("Drawing heatmaps…")
    for obj in objectives:
        draw_heatmap(data, obj, f"heatmap_{obj}.png", out_dir)

    print("Drawing barplot…")
    draw_barplot(data, out_dir)

    print("Drawing volcano plot…")
    draw_volcano(data, out_dir)

    print(f"\nDone — all figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
