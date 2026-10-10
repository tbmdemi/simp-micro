"""
Shared utility functions for the analysis notebooks in notebooks/.

Provides DOE-dataset sample loading/cleaning/plotting
(01_doe_dataset_analysis) and geometric feature extraction
(02_geometric_feature_influence).

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
from scipy import ndimage, stats

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = REPO_ROOT / "outputs" / "pipeline" / "phase1"
PHASE3_DIR = REPO_ROOT / "outputs" / "phase3"


def resolve_repo_path(p: "str | Path") -> Path:
    """Resolve a sample_path/csv_path against REPO_ROOT, independent of cwd.

    sample_path values stored in the dataset (parquet/manifest) are relative
    to the repo root. Resolving them via Path.cwd() breaks whenever the
    Jupyter kernel's working directory isn't the repo root (e.g. VS Code
    defaults to the notebook's own directory).
    """
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


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


def load_manifest_samples(manifest_path: Optional[Path] = None) -> pd.DataFrame:
    """Load manifest-style phase outputs into a notebook-friendly DataFrame.

    Ưu tiên đọc `manifest_quality.csv` (từ
    analysis/scripts/audit_convergence_and_geometry.py) thay vì
    `manifest.csv` gốc khi có sẵn - manifest_quality.csv có thêm n_iters
    THẬT (không hardcode 150) và n_components/is_connected (quét
    connectivity thật trên ảnh 64x64, xem manufacturability.check_connectivity).
    Rơi về manifest.csv (không có 2 cột trên) nếu chưa chạy script audit.
    """
    if manifest_path is None:
        quality_path = PHASE3_DIR / "manifest_quality.csv"
        manifest_path = quality_path if quality_path.exists() else PHASE3_DIR / "manifest.csv"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if df.empty:
        return df

    rename_map = {
        'v12': 'final_nu12',
        'v21': 'final_nu21',
        'volfrac_achieved': 'final_VF',
        'obj_value': 'final_obj',
        'volfrac': 'volfrac',
        'penal': 'penal',
        'rmin': 'rmin',
        'move': 'move',
        'void_size_frac': 'void_size_frac',
    }
    df = df.rename(columns=rename_map)
    df['sample_path'] = df['image_path'].apply(lambda p: str(Path(p).parent) if pd.notna(p) else np.nan)
    df['seed'] = df.get('seed', pd.Series([np.nan] * len(df)))
    df['sample_id'] = df.get('sample_id', pd.Series([np.nan] * len(df)))
    if 'n_iters' in df.columns:
        # n_iters thật từ batch_{i}_results.csv (join trong audit script) -
        # converged_flag = hội tụ SỚM vì tolerance, KHÔNG phải cột 'converged'
        # gốc (cột đó cũng =True khi chạm max_iter, xem
        # simp/core/convergence.py::ConvergenceChecker.should_stop() và
        # EXPERIMENT_LOG.md mục audit hội tụ 2026-07-24).
        df['n_iter'] = df['n_iters']
        df['converged_flag'] = (df['n_iters'] < df['max_iter']).fillna(False).astype(bool)
    else:
        # Fallback khi chưa chạy audit script: giữ hành vi cũ (kém chính
        # xác - n_iter không có sẵn trong manifest.csv gốc, và cột
        # 'converged' không phân biệt được hội tụ thật vs chạm trần).
        df['n_iter'] = 150
        df['converged_flag'] = df.get('converged', False).fillna(False).astype(bool)
    if 'is_connected' in df.columns:
        df['n_components'] = df['n_components']
        df['disconnected_flag'] = (~df['is_connected'].astype(bool)).fillna(False)
    df['void_flag'] = (df['final_VF'] < VOID_THRESHOLD).fillna(False)
    df['nu_valid_flag'] = (
        df['final_nu12'].notna() & (NU_LOWER < df['final_nu12']) & (df['final_nu12'] < NU_UPPER)
    )
    df['rotation_deg'] = 0.0
    df['mu'] = 0.0
    return df


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
    if phase1_dir is not None and Path(phase1_dir).exists():
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

    if PHASE3_DIR.exists() and (PHASE3_DIR / "manifest.csv").exists():
        # manifest_path=None -> load_manifest_samples() tự ưu tiên
        # manifest_quality.csv nếu đã chạy audit_convergence_and_geometry.py
        # (xem docstring load_manifest_samples), rơi về manifest.csv nếu chưa.
        df = load_manifest_samples(None)
        if drop_void and not df.empty:
            df = df[~df["void_flag"]].copy()
        return df

    raise FileNotFoundError("No Phase 1 samples or manifest outputs found")


# ──────────────────────────────────────────────
#  Classification
# ──────────────────────────────────────────────


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
        sample_path = row.get("sample_path")
        if sample_path is None:
            ax.text(0.5, 0.5, "No image path", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        else:
            sample_dir = resolve_repo_path(str(sample_path))
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
        seed = row.get("seed", "?")
        nu12 = row.get("final_nu12", np.nan)
        ax.set_title(
            f"{seed} | nu={nu12:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Top 10 Auxetic Designs", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  Geometric feature extraction (n_edges/thickness -> v12 influence,
#  xem notebooks/02_geometric_feature_influence.ipynb)
# ──────────────────────────────────────────────

GEOM_BIN_THRESHOLD = 0.5
GEOM_PAD = 2  # px, viền tuần hoàn quanh ảnh trước khi skeleton hóa
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Dịch ảnh nhị phân (dy, dx), lấp biên bằng 0 (nền)."""
    out = np.zeros_like(img)
    h, w = img.shape
    ys, ye = max(0, dy), h + min(0, dy)
    xs, xe = max(0, dx), w + min(0, dx)
    oys, oye = max(0, -dy), h + min(0, -dy)
    oxs, oxe = max(0, -dx), w + min(0, -dx)
    out[oys:oye, oxs:oxe] = img[ys:ye, xs:xe]
    return out


def zhang_suen_thin(binary: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen thinning, vector hóa toàn ảnh bằng numpy (không loop
    pixel). `binary`: mảng bool, 1=vật liệu. Trả skeleton bool, dày 1 pixel.
    """
    img = binary.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for sub_iter in (0, 1):
            p2 = _shift(img, -1, 0)
            p3 = _shift(img, -1, 1)
            p4 = _shift(img, 0, 1)
            p5 = _shift(img, 1, 1)
            p6 = _shift(img, 1, 0)
            p7 = _shift(img, 1, -1)
            p8 = _shift(img, 0, -1)
            p9 = _shift(img, -1, -1)

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(neighbors)  # số hàng xóm = 1
            seq = neighbors + [p2]  # A = số chuyển 0->1 theo chu trình p2..p9..p2
            A = np.zeros_like(img)
            for a, b in zip(seq[:-1], seq[1:]):
                A += ((a == 0) & (b == 1)).astype(np.uint8)

            cond_base = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if sub_iter == 0:
                cond = cond_base & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = cond_base & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def skeleton_topology(skel: np.ndarray) -> Dict[str, float]:
    """Rút gọn skeleton pixel-graph thành topology graph (contract các
    chuỗi pass-through degree=2), trả n_edges (số thanh/struts thật),
    n_junctions, n_endpoints, n_loops, n_isolated.
    """
    ys, xs = np.where(skel)
    if len(ys) == 0:
        return dict(n_edges=0, n_junctions=0, n_endpoints=0, n_loops=0, n_isolated=0)

    coords = set(zip(ys.tolist(), xs.tolist()))
    h, w = skel.shape

    def neighbors(pt):
        y, x = pt
        out = []
        for dy, dx in _NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) in coords:
                out.append((ny, nx))
        return out

    degree = {pt: len(neighbors(pt)) for pt in coords}
    keep = {pt for pt, d in degree.items() if d != 2}
    isolated = {pt for pt, d in degree.items() if d == 0}
    keep -= isolated

    visited_edges = set()
    n_edges = 0
    for start in keep:
        for nbr in neighbors(start):
            ekey = frozenset((start, nbr))
            if ekey in visited_edges:
                continue
            visited_edges.add(ekey)
            prev, cur = start, nbr
            while cur not in keep:
                nxts = [n for n in neighbors(cur) if n != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_edges.add(frozenset((cur, nxt)))
                prev, cur = cur, nxt
            if cur in keep:
                n_edges += 1

    # Vòng kín thuần túy (mọi pixel degree==2, không có junction/endpoint):
    # mỗi connected component còn lại như vậy tính là 1 "cạnh" dạng vòng.
    remaining = coords - keep - isolated
    visited_pix = set()
    n_loops = 0
    for pt in remaining:
        touched = {p for e in visited_edges for p in e}
        if pt in visited_pix or pt in touched:
            continue
        stack = [pt]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            for nbr in neighbors(cur):
                if nbr in remaining and nbr not in comp:
                    stack.append(nbr)
        if comp and not (comp & touched):
            n_loops += 1
        visited_pix |= comp

    n_junctions = sum(1 for pt in keep if degree[pt] >= 3)
    n_endpoints = sum(1 for pt in keep if degree[pt] == 1)

    return dict(
        n_edges=n_edges + n_loops,
        n_junctions=n_junctions,
        n_endpoints=n_endpoints,
        n_loops=n_loops,
        n_isolated=len(isolated),
    )


def extract_geometric_features(image: np.ndarray) -> Dict[str, float]:
    """image: (64,64) float32 [0,1] density field. Trả dict đặc trưng hình
    học: n_edges (số thanh), n_junctions, n_endpoints, độ dày trung bình
    (mean_thickness_px = 2*EDT tại pixel skeleton), n_components, solid_frac.

    Pad tuần hoàn (wrap) trước khi skeleton hóa để thanh bị cắt ngang bởi
    biên ô đơn vị không tạo endpoint giả.
    """
    binary = image > GEOM_BIN_THRESHOLD
    padded = np.pad(binary, GEOM_PAD, mode="wrap")
    skel_padded = zhang_suen_thin(padded)
    skel = skel_padded[GEOM_PAD:-GEOM_PAD, GEOM_PAD:-GEOM_PAD]

    topo = skeleton_topology(skel)

    edt_padded = ndimage.distance_transform_edt(padded)
    edt = edt_padded[GEOM_PAD:-GEOM_PAD, GEOM_PAD:-GEOM_PAD]
    skel_thickness = edt[skel] * 2.0
    mean_thickness = float(skel_thickness.mean()) if skel_thickness.size else 0.0
    std_thickness = float(skel_thickness.std()) if skel_thickness.size else 0.0

    structure = np.ones((3, 3), dtype=int)
    _, n_components = ndimage.label(binary, structure=structure)

    return {
        "n_edges": float(topo["n_edges"]),
        "n_junctions": float(topo["n_junctions"]),
        "n_endpoints": float(topo["n_endpoints"]),
        "n_isolated_specks": float(topo["n_isolated"]),
        "mean_thickness_px": mean_thickness,
        "std_thickness_px": std_thickness,
        "skeleton_length_px": float(skel.sum()),
        "n_components": float(n_components),
        "solid_frac": float(binary.mean()),
    }


def load_geometric_analysis_sample(
    n_samples: Optional[int] = 6000, seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Gộp train/val/test_ext.npz (phân tích mô tả post-hoc, không train
    model nên không có rủi ro leakage giữa split), lấy mẫu phân tầng theo
    seed_class nếu n_samples được chỉ định (None = toàn bộ ~57k mẫu).

    Trả (meta_df, images) với meta_df có cột: v12, seed_name, volfrac_param, split.
    """
    frames = []
    for split in ("train", "val", "test"):
        path = PHASE3_DIR / f"{split}_ext.npz"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        n = len(d["v12"])
        frames.append(dict(
            images=d["images"], v12=d["v12"], seed_names=d["seed_names"],
            volfrac=d["params"][:, 0], split=np.full(n, split),
        ))

    images = np.concatenate([f["images"] for f in frames], axis=0)
    v12 = np.concatenate([f["v12"] for f in frames], axis=0)
    seed_names = np.concatenate([f["seed_names"] for f in frames], axis=0)
    volfrac = np.concatenate([f["volfrac"] for f in frames], axis=0)
    split = np.concatenate([f["split"] for f in frames], axis=0)

    idx = np.arange(len(v12))
    rng = np.random.default_rng(seed)
    if n_samples is not None and n_samples < len(idx):
        df_idx = pd.DataFrame({"idx": idx, "seed_names": seed_names})
        frac = n_samples / len(idx)
        sampled = df_idx.groupby("seed_names", group_keys=False)[["idx"]].apply(
            lambda g: g.sample(n=max(1, round(len(g) * frac)), random_state=seed)
        )
        idx = rng.permutation(sampled["idx"].to_numpy())

    meta = pd.DataFrame({
        "idx": idx, "v12": v12[idx], "seed_name": seed_names[idx],
        "volfrac_param": volfrac[idx], "split": split[idx],
    })
    return meta, images[idx]


def partial_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Pearson partial correlation giữa x,y kiểm soát 1 biến `control`."""
    def resid(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx, ry = resid(x, control), resid(y, control)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_split_test(feature: np.ndarray, v12: np.ndarray) -> Dict[str, float]:
    """Chia theo median thành nhóm cao/thấp, Mann-Whitney U + rank-biserial
    effect size - đúng yêu cầu "câu nhị phân" đánh giá ảnh hưởng."""
    med = np.median(feature)
    low_mask, high_mask = feature <= med, feature > med
    low, high = v12[low_mask], v12[high_mask]
    if len(low) < 5 or len(high) < 5:
        return dict(median=float(med), n_low=len(low), n_high=len(high),
                    u_stat=float("nan"), p_value=float("nan"), rank_biserial=float("nan"),
                    mean_v12_low=float(np.mean(low)) if len(low) else float("nan"),
                    mean_v12_high=float(np.mean(high)) if len(high) else float("nan"))
    u_stat, p_value = stats.mannwhitneyu(low, high, alternative="two-sided")
    rank_biserial = 1 - (2 * u_stat) / (len(low) * len(high))
    return dict(median=float(med), n_low=int(low_mask.sum()), n_high=int(high_mask.sum()),
                u_stat=float(u_stat), p_value=float(p_value), rank_biserial=float(rank_biserial),
                mean_v12_low=float(np.mean(low)), mean_v12_high=float(np.mean(high)))


GEOM_FEATURE_COLS = ["n_edges", "n_junctions", "n_endpoints", "mean_thickness_px",
                      "std_thickness_px", "n_components", "solid_frac"]


def analyze_geometric_group(df: pd.DataFrame, label: str) -> Dict:
    """Tương quan Pearson/Spearman + partial corr (kiểm soát volfrac) +
    Mann-Whitney nhị phân + hồi quy đa biến chuẩn hóa, cho 1 nhóm mẫu
    (pooled hoặc 1 seed_class)."""
    v12 = df["v12"].to_numpy()
    volfrac = df["volfrac_param"].to_numpy()
    out = {"label": label, "n": len(df), "features": {}}
    for col in GEOM_FEATURE_COLS:
        x = df[col].to_numpy()
        if np.std(x) == 0 or len(df) < 8:
            continue
        pear_r, pear_p = stats.pearsonr(x, v12)
        spear_r, spear_p = stats.spearmanr(x, v12)
        pcorr = partial_corr(x, v12, volfrac) if col != "solid_frac" else float("nan")
        out["features"][col] = {
            "pearson_r": float(pear_r), "pearson_p": float(pear_p),
            "spearman_r": float(spear_r), "spearman_p": float(spear_p),
            "partial_corr_ctrl_volfrac": pcorr,
            "binary_test": binary_split_test(x, v12),
        }

    all_reg_cols = ["n_edges", "mean_thickness_px", "n_components", "volfrac_param"]
    reg_cols = [c for c in all_reg_cols if df[c].std() > 0]
    X = df[reg_cols].to_numpy(dtype=float)
    if len(df) > len(reg_cols) + 5 and len(reg_cols) >= 2:
        Xz = (X - X.mean(axis=0)) / X.std(axis=0)
        Xz1 = np.column_stack([Xz, np.ones(len(df))])
        coef, *_ = np.linalg.lstsq(Xz1, v12, rcond=None)
        pred = Xz1 @ coef
        ss_res, ss_tot = float(np.sum((v12 - pred) ** 2)), float(np.sum((v12 - v12.mean()) ** 2))
        out["regression"] = {
            "cols": reg_cols,
            "standardized_coef": {c: float(k) for c, k in zip(reg_cols, coef[:-1])},
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        }
    return out