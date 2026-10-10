"""
Analysis pipeline for SIMP topology optimization results.

Provides utilities for analyzing convergence data, image quality,
and generating HTML reports from SIMP optimization runs.

Core API:
    dataset:    Dataset overview and convergence analysis.
    image:      Image quality metrics (binary rate, edge density, etc.).
    report:     HTML report generation.
    cli:        Command-line interface for the analysis pipeline.

Standalone scripts live in ``analysis.scripts`` - run them via::

    python -m analysis.scripts.plot_correlation_figures
"""

from ._version import __version__, VERSION_INFO
from . import dataset
from . import image
from . import report
from . import cli

__all__ = ['__version__', 'VERSION_INFO', 'dataset', 'image', 'report', 'cli']
