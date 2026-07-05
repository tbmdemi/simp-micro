"""
Visualization helpers for sensitivity analysis results.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_src(
    src_result: Dict,
    figsize: Tuple[float, float] = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Vẽ bar plot SRC coefficients.

    Args:
        src_result: Dict từ regression.compute_src().
        figsize: Kích thước figure.
        ax: Axes có sẵn (nếu không, tạo mới).

    Returns:
        Figure object.
    """
    coef = src_result['coef']
    params = list(coef.keys())
    values = list(coef.values())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = ['#e74c3c' if v < 0 else '#2980b9' for v in values]
    bars = ax.barh(params, values, color=colors, edgecolor='white', height=0.6)

    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Standardized Regression Coefficient (SRC)', fontsize=11)
    ax.set_title('Parameter Sensitivity - SRC', fontsize=13, fontweight='bold')

    r2 = src_result.get('r2', None)
    if r2 is not None:
        ax.text(0.98, 0.02, f'R² = {r2:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    fig.tight_layout()
    return fig


def plot_sobol(
    sobol_result: Dict,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Vẽ grouped bar plot Sobol S1 và ST.

    Args:
        sobol_result: Dict từ sobol.compute_sobol_from_surrogate().
        figsize: Kích thước figure.

    Returns:
        Figure object.
    """
    params = sobol_result['param_names']
    s1 = [sobol_result['S1'][p] or 0.0 for p in params]
    st = [sobol_result['ST'][p] or 0.0 for p in params]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(params))
    width = 0.35

    bars1 = ax.bar(x - width / 2, s1, width, label='S₁ (first-order)',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width / 2, st, width, label='ST (total-order)',
                   color='#e74c3c', edgecolor='white', alpha=0.85)

    ax.set_xlabel('Parameter', fontsize=11)
    ax.set_ylabel('Sobol Index', fontsize=11)
    ax.set_title('Sobol Sensitivity Indices', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    r2 = sobol_result.get('r2_surrogate', None)
    if r2 is not None:
        ax.text(0.98, 0.98, f'Surrogate R² = {r2:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    fig.tight_layout()
    return fig


def plot_anova(
    anova_result: Dict,
    figsize: Tuple[float, float] = (8, 5),
    alpha: float = 0.05,
) -> plt.Figure:
    """Vẽ F-statistics từ ANOVA với significance threshold.

    Args:
        anova_result: Dict từ anova.compute_anova_from_csv().
        figsize: Kích thước figure.
        alpha: Mức ý nghĩa.

    Returns:
        Figure object.
    """
    oneway = anova_result['oneway']
    params = list(oneway.keys())
    f_vals = []
    p_vals = []

    for p in params:
        info = oneway[p]
        f_vals.append(info['F'] if info['F'] is not None else 0.0)
        p_vals.append(info['p'] if info['p'] is not None else 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] + 2, figsize[1]))

    # Bar plot F-statistics
    colors = ['#27ae60' if pv < alpha else '#bdc3c7' for pv in p_vals]
    ax1.barh(params, f_vals, color=colors, edgecolor='white', height=0.6)
    ax1.set_xlabel('F-statistic', fontsize=11)
    ax1.set_title('ANOVA - F Statistics', fontsize=12, fontweight='bold')
    ax1.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    # -log10(p-value)
    neg_log_p = [-np.log10(pv + 1e-300) for pv in p_vals]
    threshold_line = -np.log10(alpha)
    ax2.barh(params, neg_log_p, color='#8e44ad', edgecolor='white', height=0.6)
    ax2.axvline(threshold_line, color='red', linewidth=1.0, linestyle='--',
                label=f'α = {alpha}')
    ax2.set_xlabel(r'$-\log_{10}(p\text{-value})$', fontsize=11)
    ax2.set_title('ANOVA - Significance', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_parameter_classification(
    classification: Dict[str, Dict],
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Vẽ scatter plot SRC vs Sobol ST, tô màu theo phân loại.

    Args:
        classification: Dict từ classify.classify_parameters().
        figsize: Kích thước figure.

    Returns:
        Figure object.
    """
    class_colors = {
        'highly_sensitive': '#e74c3c',
        'moderately_sensitive': '#f39c12',
        'locally_sensitive': '#3498db',
        'not_sensitive': '#95a5a6',
        'no_data': '#bdc3c7',
    }
    class_labels = {
        'highly_sensitive': 'Highly sensitive',
        'moderately_sensitive': 'Moderately',
        'locally_sensitive': 'Local only',
        'not_sensitive': 'Not sensitive',
        'no_data': 'No data',
    }

    fig, ax = plt.subplots(figsize=figsize)

    for p, info in classification.items():
        src = info['src'] if info['src'] is not None else 0.0
        st = info['st'] if info['st'] is not None else 0.0
        cls = info['class']
        color = class_colors.get(cls, '#95a5a6')

        ax.scatter(src, st, c=color, s=100, edgecolors='white',
                   linewidth=0.8, zorder=5)
        ax.annotate(p, (src, st), textcoords='offset points',
                    xytext=(8, 4), fontsize=9, alpha=0.85)

    # Legend
    handles = []
    for cls, color in class_colors.items():
        if any(info['class'] == cls for info in classification.values()):
            from matplotlib.patches import Patch
            handles.append(Patch(color=color, label=class_labels[cls]))
    ax.legend(handles=handles, fontsize=9, loc='best')

    ax.axhline(0.2, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axvline(0.1, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.set_xlabel('SRC (standardized regression coefficient)', fontsize=11)
    ax.set_ylabel('Sobol ST (total-order index)', fontsize=11)
    ax.set_title('Parameter Classification by Sensitivity', fontsize=13, fontweight='bold')

    fig.tight_layout()
    return fig