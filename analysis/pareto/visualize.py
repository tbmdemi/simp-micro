"""
Visualization for Pareto Front
================================
Vẽ Pareto front 2D và 3D từ dữ liệu multi-objective.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_pareto_front_2d(
    df_all: 'pd.DataFrame',
    obj_cols: List[str],
    frontier_mask: np.ndarray,
    figsize: Tuple[float, float] = (10, 7),
    maximize: bool = False,
    color_all: str = '#95a5a6',
    color_frontier: str = '#e74c3c',
    alpha_all: float = 0.4,
) -> plt.Figure:
    """Vẽ Pareto front 2D.

    Args:
        df_all: DataFrame toàn bộ dữ liệu.
        obj_cols: 2 cột objective.
        frontier_mask: Boolean mask đánh dấu Pareto points.
        figsize: Kích thước figure.
        maximize: True nếu maximize objectives.
        color_all: Màu cho non-Pareto points.
        color_frontier: Màu cho Pareto points.
        alpha_all: Độ trong suốt cho non-Pareto.

    Returns:
        Figure object.
    """
    if len(obj_cols) != 2:
        raise ValueError('plot_pareto_front_2d requires exactly 2 objectives')

    fig, ax = plt.subplots(figsize=figsize)

    # Non-Pareto
    non_frontier = ~frontier_mask
    if non_frontier.sum() > 0:
        ax.scatter(
            df_all.loc[non_frontier, obj_cols[0]],
            df_all.loc[non_frontier, obj_cols[1]],
            c=color_all, alpha=alpha_all, s=20, edgecolors='none',
            label='Non-Pareto',
        )

    # Pareto front
    frontier_df = df_all[frontier_mask].sort_values(obj_cols[0])
    ax.scatter(
        frontier_df[obj_cols[0]],
        frontier_df[obj_cols[1]],
        c=color_frontier, s=50, edgecolors='white', linewidth=0.5,
        label=f'Pareto front ({len(frontier_df)} points)',
        zorder=5,
    )
    # Đường nối
    ax.plot(
        frontier_df[obj_cols[0]],
        frontier_df[obj_cols[1]],
        color=color_frontier, linewidth=1.0, alpha=0.6, linestyle='--',
    )

    ax.set_xlabel(obj_cols[0], fontsize=11)
    ax.set_ylabel(obj_cols[1], fontsize=11)
    ax.set_title('Pareto Front', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_pareto_front_3d(
    df_all: 'pd.DataFrame',
    obj_cols: List[str],
    frontier_mask: np.ndarray,
    figsize: Tuple[float, float] = (10, 8),
    maximize: bool = False,
) -> plt.Figure:
    """Vẽ Pareto front 3D.

    Args:
        df_all: DataFrame toàn bộ dữ liệu.
        obj_cols: 3 cột objective.
        frontier_mask: Boolean mask đánh dấu Pareto points.
        figsize: Kích thước figure.
        maximize: True nếu maximize objectives.

    Returns:
        Figure object.
    """
    if len(obj_cols) != 3:
        raise ValueError('plot_pareto_front_3d requires exactly 3 objectives')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Non-Pareto
    non_frontier = ~frontier_mask
    if non_frontier.sum() > 0:
        ax.scatter(
            df_all.loc[non_frontier, obj_cols[0]],
            df_all.loc[non_frontier, obj_cols[1]],
            df_all.loc[non_frontier, obj_cols[2]],
            c='#95a5a6', alpha=0.3, s=10,
            label='Non-Pareto',
        )

    # Pareto front
    frontier_df = df_all[frontier_mask]
    ax.scatter(
        frontier_df[obj_cols[0]],
        frontier_df[obj_cols[1]],
        frontier_df[obj_cols[2]],
        c='#e74c3c', s=40, edgecolors='white', linewidth=0.3,
        label=f'Pareto front ({len(frontier_df)} points)',
    )

    ax.set_xlabel(obj_cols[0], fontsize=10)
    ax.set_ylabel(obj_cols[1], fontsize=10)
    ax.set_zlabel(obj_cols[2], fontsize=10)
    ax.set_title('Pareto Front (3D)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_pareto_by_seed(
    df: 'pd.DataFrame',
    obj_cols: List[str],
    seed_col: str = 'seed',
    figsize: Tuple[float, float] = (12, 8),
    maximize: bool = False,
) -> plt.Figure:
    """Vẽ Pareto front, tô màu theo seed.

    Args:
        df: DataFrame dữ liệu.
        obj_cols: 2 cột objective.
        seed_col: Tên cột seed.
        figsize: Kích thước figure.
        maximize: True nếu maximize.

    Returns:
        Figure object.
    """
    seeds = df[seed_col].unique() if seed_col in df.columns else ['all']

    fig, ax = plt.subplots(figsize=figsize)
    cmap = cm.get_cmap('tab10', len(seeds))

    for i, seed in enumerate(seeds):
        if seed_col in df.columns:
            sub = df[df[seed_col] == seed]
        else:
            sub = df

        ax.scatter(
            sub[obj_cols[0]], sub[obj_cols[1]],
            c=[cmap(i)], s=20, alpha=0.6, edgecolors='none',
            label=str(seed),
        )

    ax.set_xlabel(obj_cols[0], fontsize=11)
    ax.set_ylabel(obj_cols[1], fontsize=11)
    ax.set_title('Pareto Front by Seed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, title='Seed')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig