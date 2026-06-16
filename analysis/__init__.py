"""
Analysis pipeline for SIMP topology optimization results.

Provides utilities for analyzing convergence data, image quality,
and generating HTML reports from SIMP optimization runs.

Core API:
    dataset:    Dataset overview and convergence analysis.
    image:      Image quality metrics (binary rate, edge density, etc.).
    report:     HTML report generation.
    cli:        Command-line interface for the analysis pipeline.

Standalone scripts live in ``analysis.scripts`` — run them via::

    python -m analysis.scripts.aggregate_correlations
    python -m analysis.scripts.plot_correlation_figures
    python -m analysis.scripts.select_representative
"""

from ._version import __version__, VERSION_INFO
from . import dataset
from . import image
from . import report
from . import cli

# Utility functions
from .utils import safe_float, round_metric, resolve_phase1_dir, load_json

__all__ = [
    'dataset', 'image', 'report', 'cli',
    'safe_float', 'round_metric', 'resolve_phase1_dir', 'load_json',
]
