"""
Shared utility functions for the SIMP analysis pipeline.

Provides safe numeric conversion, path resolution, JSON I/O,
and other helpers reused across modules and scripts.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def safe_float(value: Any, fallback: float = float('nan')) -> float:
    """Convert a value to float, returning *fallback* on failure.

    Args:
        value: Input to convert.
        fallback: Default value if conversion fails.

    Returns:
        Float value.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def safe_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Convert a value to int, returning *fallback* on failure.

    Args:
        value: Input to convert.
        fallback: Default value if conversion fails.

    Returns:
        Int or fallback.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def round_metric(value: float, decimals: int = 4) -> Optional[float]:
    """Round a float metric to *decimals* places, returning None if NaN.

    Args:
        value: Float value to round.
        decimals: Number of decimal places.

    Returns:
        Rounded float, or None if NaN.
    """
    if np.isnan(value):
        return None
    return round(value, decimals)


def resolve_phase1_dir(base_dir: Union[str, Path]) -> Path:
    """Resolve the Phase 1 output directory.

    Checks the given path and a few common relative alternatives.

    Lưu ý: Hàm này hiện không được sử dụng trong analysis modules.
    Giữ lại để tương thích ngược (API public).

    Args:
        base_dir: Suggested directory path.

    Returns:
        Path to the Phase 1 directory.
    """
    candidates = [
        Path(base_dir),
        Path('outputs/pipeline/phase1'),
        Path('.').resolve() / 'outputs' / 'pipeline' / 'phase1',
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path(base_dir)


def load_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on failure.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict, or None.
    """
    path = Path(path)
    if not path.exists():
        logger.warning('JSON file not found: %s', path)
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning('Failed to load JSON %s: %s', path, exc)
        return None
