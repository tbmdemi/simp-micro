# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-05-21

### Added
- `analysis/` module: new structured analysis pipeline replacing `src/`.
  - `analysis/dataset.py`: Dataset overview, convergence metrics, auxetic classification.
  - `analysis/image.py`: Image quality metrics (binary rate, edge density, noise, symmetry).
  - `analysis/report.py`: Self-contained HTML report generation.
  - `analysis/cli.py`: CLI interface for analysis commands.
- `simp/core/convergence.py`: Dedicated `ConvergenceChecker` class with design change and objective stability criteria.
- `pyproject.toml`: Standard Python packaging with setuptools.
- `.gitignore`: Comprehensive Python project ignore rules.
- `requirements-core.txt`, `requirements-analysis.txt`, `requirements-dev.txt`: Split dependency files.
- `Makefile`: Common commands (install, test, lint, format, run).
- `CHANGELOG.md`: This file.

### Changed
- **`simp/io/logger.py`**: Buffered CSV writing for performance (configurable buffer size).
- **`simp/io/visualizer.py`**: Full docstrings, type hints, cleaner API.
- **`simp/core/solver.py`**: Reduced sparse↔dense conversions; all submatrix operations stay in sparse format.
- **`simp/config.py`**: Full `SimpConfig` dataclass with `__post_init__` validation.
- **`simp/runner.py`**: Integrated `ConvergenceChecker`; cleaner loop structure; returns results dict.
- **`simp/main.py`**: Added `--version`, `--list-seeds`, `--verbose` flags; seed registry; error handling.
- **`simp/__init__.py`**: Added `__version__`, `__author__`, `__license__`.
- **`simp/core/__init__.py`**: Exports `ConvergenceChecker`.
- **`simp/io/__init__.py`**: Clean `__all__` exports.

### Removed
- `src/` directory (replaced by `analysis/` module).
- `requirements.txt` (split into `requirements-*.txt` files).

### Fixed
- Logger no longer opens file on every `log()` call (buffered I/O).
- Solver no longer converts sparse submatrices to dense unnecessarily.
- Config validation catches invalid parameters early.

## [1.0.0] — 2026-04-xx

### Added
- Initial SIMP topology optimization engine.
- Core modules: fem, filter, pbc, solver, oc.
- Material properties, homogenization, objective functions.
- Seed patterns: circle, square, hourglass, four_circle, hexagonal, nine_circle, cross, grid_voids, small_cross, half_circle.
- CSV logging and PNG visualization.
- HTML workflow documentation (workflow.html, workflow_vi.html).
- MATLAB reference implementations in `data/`.
- Analysis notebooks in `notebooks/`.
