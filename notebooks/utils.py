"""
Shared utility functions for Phase 1 data analysis notebooks.

Provides loading, cleaning, classification, and metric computation
for the SIMP auxetic optimization pipeline results.

Usage:
    from utils import load_all_samples, plot_convergence
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
PHASE1_DIR = Path("../outputs/pipeline/phase1")
SEEDS = [
    "circle", "square", "hourglass", "four_circle", "hexagonal",
    "nine_circle", "cross_rectangular", "grid_circular_voids",
    "small_square_cross", "circle_half_quarter",
]
MAX_ITER = 150   # from pipeline/params.py FIXED_PARAMS
VOID_THRESHOLD = 0.01  # Volume_Fraction below this → void collapse
NU_LOWER = -1.0   # physically plausible range for Poisson's ratio
NU_UPPER = 0.5

# ──────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────


def scan_phase1_samples(
    phase1_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Scan all seed/sample directories and return metadata for each.

    Returns a list of dicts with:
        seed, sample_id, sample_path, metadata_path, csv_path, image_paths
    """
    if phase1_dir is None:
        phase1_dir = PHASE1_DIR
    phase1_dir = Path(phase1_dir)
    if not phase1_dir.exists():
        raise FileNotFoundError(f"Phase 1 directory not found: {phase1_dir}")

    samples: List[Dict[str, Any]] = []
    # Each subdirectory is a seed name
    for seed_dir in sorted(phase1_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed_name = seed_dir.name
        # Each sample_XXXX directory
        for sample_dir in sorted(seed_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id_str = sample_dir.name  # e.g. "sample_0000"
            csv_path = sample_dir / "iteration_data.csv"
            meta_path = sample_dir / "metadata.json"

            # Gather image paths (final iteration snapshot)
            images = sorted(sample_dir.glob("iteration_*.png"))

            if csv_path.exists():
                samples.append({
                    "seed": seed_name,
                    "sample_id": sample_id_str,
                    "sample_path": sample_dir,
                    "csv_path": csv_path,
                    "metadata_path": meta_path if meta_path.exists() else None,
                    "image_paths": images,
                })
    return samples


def load_sample_last_row(csv_path: Path) -> Optional[pd.Series]:
    """Load the last row of an iteration_data.csv.

    Returns a Series with columns:
        Iteration, Poisson_v12, Poisson_v21, Objective, Volume_Fraction
    or None on failure.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df.iloc[-1]
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None


def load_sample_history(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load the full iteration history from iteration_data.csv.

    Returns a DataFrame, or None on failure.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        # Ensure numeric columns
        for col in ["Poisson_v12", "Poisson_v21", "Objective", "Volume_Fraction"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None


def load_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata.json as a dict, or None on failure."""
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", meta_path, exc)
        return None


def load_all_samples(
    phase1_dir: Optional[Path] = None,
    drop_void: bool = False,
) -> pd.DataFrame:
    """Build a master DataFrame from all Phase 1 samples.

    Each row corresponds to one sample, with columns:
        sample_path, seed, sample_id,
        final_nu12, final_nu21, final_VF, final_obj, n_iter,
        converged_flag, void_flag,
        volfrac, penal, rmin, move, void_size_frac, rotation_deg, mu,
        nu_valid_flag

    Args:
        phase1_dir: Path to Phase 1 output directory.
        drop_void: If True, drop rows where void_flag is True.

    Returns:
        pd.DataFrame with one row per sample.
    """
    samples = scan_phase1_samples(phase1_dir)
    rows = []
    for s in samples:
        last = load_sample_last_row(s["csv_path"])
        meta = load_metadata(s["metadata_path"]) if s["metadata_path"] else None

        row: Dict[str, Any] = {
            "sample_path": str(s["sample_path"]),
            "seed": s["seed"],
            "sample_id": s["sample_id"],
        }

        # --- Last-row metrics ---
        if last is not None:
            row["final_nu12"] = last.get("Poisson_v12", np.nan)
            row["final_nu21"] = last.get("Poisson_v21", np.nan)
            row["final_obj"] = last.get("Objective", np.nan)
            row["final_VF"] = last.get("Volume_Fraction", np.nan)
            row["n_iter"] = int(last.get("Iteration", 0))
        else:
            row["final_nu12"] = np.nan
            row["final_nu21"] = np.nan
            row["final_obj"] = np.nan
            row["final_VF"] = np.nan
            row["n_iter"] = 0

        # --- Flags ---
        row["converged_flag"] = bool(
            not np.isnan(row["n_iter"]) and row["n_iter"] < MAX_ITER - 1
        )
        row["void_flag"] = bool(
            not np.isnan(row["final_VF"]) and row["final_VF"] < VOID_THRESHOLD
        )
        row["nu_valid_flag"] = bool(
            not np.isnan(row["final_nu12"])
            and NU_LOWER < row["final_nu12"] < NU_UPPER
        )

        # --- Parameters from metadata ---
        if meta and "params" in meta:
            params = meta["params"]
            row["volfrac"] = params.get("volfrac", np.nan)
            row["penal"] = params.get("penal", np.nan)
            row["rmin"] = params.get("rmin", np.nan)
            row["move"] = params.get("move", np.nan)
            row["void_size_frac"] = params.get("void_size_frac", np.nan)
            row["rotation_deg"] = params.get("rotation_deg", np.nan)
            row["mu"] = params.get("mu", np.nan)
        else:
            for k in ("volfrac", "penal", "rmin", "move", "void_size_frac",
                      "rotation_deg", "mu"):
                row[k] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if drop_void and not df.empty:
        df = df[~df["void_flag"]].copy()
    return df


# ──────────────────────────────────────────────
#  Classification
# ──────────────────────────────────────────────


def classify_void(vf: float, threshold: float = VOID_THRESHOLD) -> str:
    """Classify a sample as VOID or OK based on Volume_Fraction."""
    if np.isnan(vf):
        return "UNKNOWN"
    return "VOID" if vf < threshold else "OK"


def classify_auxetic_quality(nu12: float) -> str:
    """Classify auxetic performance into categories.

    Args:
        nu12: Final Poisson ratio nu12.

    Returns:
        'Strong auxetic' (nu < -0.5),
        'Moderate auxetic' (-0.5 <= nu < -0.3),
        'Weak auxetic' (-0.3 <= nu < 0),
        'Not auxetic' (nu >= 0),
        or 'INVALID' (NaN).
    """
    if np.isnan(nu12):
        return "INVALID"
    if nu12 < -0.5:
        return "Strong auxetic"
    elif nu12 < -0.3:
        return "Moderate auxetic"
    elif nu12 < 0.0:
        return "Weak auxetic"
    else:
        return "Not auxetic"


# ──────────────────────────────────────────────
#  Metric Helpers
# ──────────────────────────────────────────────


def compute_coupling_ratio(
    Q: np.ndarray, eps: float = 1e-12
) -> float:
    """Compute shear-normal coupling ratio |Q13|/sqrt(Q11*Q22).

    High coupling (>1e-3) indicates significant rotation effect.
    Requires Q tensor (3x3) in Voigt notation [11, 22, 12].
    """
    scale = np.sqrt(max(Q[0, 0] * Q[1, 1], eps))
    return float(max(abs(Q[0, 2]), abs(Q[1, 2])) / scale)


def safe_spearman(df: pd.DataFrame, col1: str, col2: str) -> Tuple[float, float]:
    """Compute Spearman correlation and p-value, returning NaN on failure."""
    from scipy.stats import spearmanr
    clean = df[[col1, col2]].dropna()
    if len(clean) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(clean[col1], clean[col2])
    return float(r), float(p)


# ──────────────────────────────────────────────
#  Plotting Helpers
# ──────────────────────────────────────────────


def plot_convergence(
    ax: plt.Axes,
    history: pd.DataFrame,
    label_prefix: str = "",
    color_nu: str = "#1f77b4",
    color_obj: str = "#d62728",
) -> None:
    """Plot nu12 and Objective convergence on a given Axes.

    Args:
        ax: Matplotlib Axes to draw on.
        history: DataFrame with columns Iteration, Poisson_v12, Objective.
        label_prefix: Optional prefix for legend labels.
        color_nu: Color for nu12 line.
        color_obj: Color for Objective line.
    """
    iters = history["Iteration"].values
    nu12 = history["Poisson_v12"].values
    obj = history["Objective"].values

    ax_twin = ax.twinx()
    line1 = ax.plot(iters, nu12, color=color_nu, lw=1.5,
                    label=f"{label_prefix}nu12" if label_prefix else "nu12")
    line2 = ax_twin.plot(iters, obj, color=color_obj, lw=1.5, alpha=0.7,
                         label=f"{label_prefix}Objective" if label_prefix else "Objective")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("nu12", color=color_nu)
    ax.tick_params(axis="y", labelcolor=color_nu)
    ax_twin.set_ylabel("Objective", color=color_obj)
    ax_twin.tick_params(axis="y", labelcolor=color_obj)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")


def plot_top10_grid(
    top10: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 6),
) -> plt.Figure:
    """Display a grid of final-iteration images for the top 10 designs.

    Expects top10 to have columns: sample_path, seed, final_nu12.

    Returns:
        matplotlib Figure.
    """
    n = len(top10)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten() if rows > 1 else axes if cols > 1 else [axes]

    for i, (_, row) in enumerate(top10.iterrows()):
        ax = axes_flat[i]
        sample_dir = Path(row["sample_path"])
        # Find the last iteration image
        images = sorted(sample_dir.glob("iteration_*.png"))
        img_path = images[-1] if images else None
        if img_path and img_path.exists():
            from PIL import Image
            img = Image.open(img_path)
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "No image", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(
            f"{row['seed']} | nu={row['final_nu12']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Top 10 Auxetic Designs", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig