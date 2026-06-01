"""
Analysis pipeline for SIMP topology optimization results.

Provides utilities for analyzing convergence data, image quality,
and generating HTML reports from SIMP optimization runs.

Modules:
    dataset:    Dataset overview and convergence analysis.
    image:      Image quality metrics (binary rate, edge density, etc.).
    report:     HTML report generation.
    cli:        Command-line interface for the analysis pipeline.
"""

__version__ = '1.1.0'

__all__ = ['dataset', 'image', 'report', 'cli']
