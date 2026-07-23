"""
Sampling strategies for adaptive multi-batch design of experiments.

Provides:
  - Sobol sequence sampling (low-discrepancy, quasi-random)
  - Optimized Latin Hypercube Sampling (LHS) with max-min criterion
  - Parameter space transformations (normalized ↔ physical)
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


def _validate_ranges(
    active_params: Dict[str, Dict[str, List[float]]],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Extract parameter names, lower bounds, and upper bounds from config.

    Args:
        active_params: Dict mapping param name -> {'range': [low, high]}

    Returns:
        Tuple of (param_names, lb, ub) where lb/ub are 1D arrays.

    Raises:
        ValueError: If any parameter range is invalid.
    """
    param_names = sorted(active_params.keys())
    lb = np.zeros(len(param_names))
    ub = np.zeros(len(param_names))

    for i, name in enumerate(param_names):
        rng = active_params[name]["range"]
        if len(rng) != 2 or rng[0] >= rng[1]:
            raise ValueError(
                f"Invalid range for '{name}': {rng}. Must be [low, high] with low < high."
            )
        lb[i] = rng[0]
        ub[i] = rng[1]

    return param_names, lb, ub


def _sobol_sequence(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate Sobol low-discrepancy sequence using native Sobol' implementation.

    Falls back to random if scipy is not available.

    Args:
        n: Number of samples.
        d: Number of dimensions.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n, d) with values in [0, 1].
    """
    try:
        from scipy.stats.qmc import Sobol

        # Round n up to nearest power of 2 to suppress scipy warning,
        # then truncate back to the requested number.
        n_pow2 = 1
        while n_pow2 < n:
            n_pow2 <<= 1

        sobol = Sobol(d, scramble=True, seed=seed)
        samples = sobol.random(n_pow2)
        return samples[:n]  # truncate to exact n
    except ImportError:
        warnings.warn(
            "scipy.stats.qmc not available. Falling back to random sampling."
        )
        rng = np.random.default_rng(seed)
        return rng.random((n, d))


def _optimized_lhs(
    n: int, d: int, seed: Optional[int] = None, n_iter: int = 100
) -> np.ndarray:
    """Generate an optimized Latin Hypercube Sample using the max-min criterion.

    Uses random search over LHS permutations to maximise the minimum
    distance between points (space-filling property).

    Args:
        n: Number of samples.
        d: Number of dimensions.
        seed: Random seed.
        n_iter: Number of optimisation iterations.

    Returns:
        Array of shape (n, d) with values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    def _lhs_sample() -> np.ndarray:
        """Generate a basic LHS sample."""
        sample = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            sample[:, j] = (perm + rng.random(n)) / n
        return sample

    def _min_dist(sample: np.ndarray) -> float:
        """Compute minimum pairwise Euclidean distance."""
        best = float("inf")
        for i in range(n):
            diffs = sample[i + 1 :] - sample[i]
            sq = np.sum(diffs * diffs, axis=1)
            min_sq = np.min(sq) if sq.size > 0 else float("inf")
            if min_sq < best:
                best = min_sq
        return np.sqrt(best)

    # Initial sample
    best_sample = _lhs_sample()
    best_min = _min_dist(best_sample)

    for _ in range(n_iter):
        candidate = _lhs_sample()
        d_min = _min_dist(candidate)
        if d_min > best_min:
            best_min = d_min
            best_sample = candidate

    return best_sample


def _normalize_to_physical(
    samples_norm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Map normalised samples in [0,1]^d to physical parameter space.

    Args:
        samples_norm: Array of shape (n, d) in [0, 1].
        lb: Lower bounds, shape (d,).
        ub: Upper bounds, shape (d,).

    Returns:
        Array of shape (n, d) in physical units.
    """
    return lb + samples_norm * (ub - lb)


def generate_samples(
    n: int,
    active_params: Dict[str, Dict[str, List[float]]],
    strategy: str = "sobol",
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, List[float]]:
    """Generate parameter samples using the specified strategy.

    Args:
        n: Number of samples to generate.
        active_params: Dict mapping param name -> {'range': [low, high]}.
        strategy: One of 'sobol', 'lhs', 'random'.
        seed: Random seed.
        **kwargs: Additional arguments passed to the sampler.

    Returns:
        Dict mapping parameter name -> list of sampled values.

    Raises:
        ValueError: If strategy is unknown or n is invalid.

    Example:
        >>> params = {
        ...     "volfrac": {"range": [0.3, 0.7]},
        ...     "void_size_frac": {"range": [0.1, 0.4]},
        ... }
        >>> samples = generate_samples(100, params, strategy="lhs", seed=42)
        >>> len(samples["volfrac"])
        100
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    param_names, lb, ub = _validate_ranges(active_params)
    d = len(param_names)

    if strategy == "sobol":
        samples_norm = _sobol_sequence(n, d, seed=seed)
    elif strategy in ("lhs", "optimized_lhs"):
        n_iter = kwargs.get("n_iter", 100)
        samples_norm = _optimized_lhs(n, d, seed=seed, n_iter=n_iter)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        samples_norm = rng.random((n, d))
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: 'sobol', 'lhs', 'optimized_lhs', 'random'."
        )

    # Clip to [0, 1] to avoid numerical overshoot
    samples_norm = np.clip(samples_norm, 0.0, 1.0)
    samples_phys = _normalize_to_physical(samples_norm, lb, ub)

    result: Dict[str, List[float]] = {}
    for i, name in enumerate(param_names):
        result[name] = [round(float(v), 6) for v in samples_phys[:, i]]

    return result


def generate_samples_dataframe(
    n: int,
    active_params: Dict[str, Dict[str, List[float]]],
    strategy: str = "sobol",
    seed: Optional[int] = None,
    **kwargs,
) -> "pd.DataFrame":
    """Generate samples and return as a pandas DataFrame.

    Args:
        n: Number of samples.
        active_params: Dict mapping param name -> {'range': [low, high]}.
        strategy: Sampling strategy ('sobol', 'lhs', 'random').
        seed: Random seed.
        **kwargs: Additional arguments passed to the sampler.

    Returns:
        DataFrame with one column per parameter and one row per sample.
    """
    import pandas as pd

    samples_dict = generate_samples(n, active_params, strategy, seed, **kwargs)
    return pd.DataFrame(samples_dict)


def generate_design(
    n_samples: int,
    param_ranges: Dict[str, Tuple[float, float]],
    strategy: "SamplingStrategy",
    batch_id: int,
    seed_map: List[str],
    objective_map: List[str],
) -> "pd.DataFrame":
    """Generate a full factorial design DataFrame across seeds x objectives x samples.

    For each (seed, objective) combination, ``n_samples`` parameter sets are drawn
    from the given strategy. The resulting DataFrame has one row per combination,
    with columns for parameters, ``seed``, ``objective``, and ``batch_id``.

    Args:
        n_samples: Number of parameter samples per (seed, objective) pair.
        param_ranges: Dict mapping param name -> (low, high).
        strategy: SamplingStrategy enum value.
        batch_id: Batch identifier for tracking.
        seed_map: List of seed shape names.
        objective_map: List of objective function names.

    Returns:
        DataFrame with columns: seed, objective, batch_id, *params.

    Example:
        >>> pr = {"E0": (200, 400), "nu": (-0.5, 0.5)}
        >>> df = generate_design(10, pr, SamplingStrategy.SOBOL, 1,
        ...                      ["circle", "square"], ["auxetic"])
        >>> len(df)
        20  # 2 seeds x 1 objective x 10 samples
        >>> list(df.columns)
        ["seed", "objective", "batch_id", "E0", "nu"]
    """
    import pandas as pd

    # Convert param_ranges to the format generate_samples expects
    active_params: Dict[str, Dict[str, List[float]]] = {}
    for pname, (lo, hi) in param_ranges.items():
        active_params[pname] = {"range": [float(lo), float(hi)]}

    # Map SamplingStrategy enum to string
    strategy_str = strategy.value if hasattr(strategy, "value") else str(strategy)

    rows = []
    for seed in seed_map:
        for obj in objective_map:
            samples_dict = generate_samples(
                n=n_samples,
                active_params=active_params,
                strategy=strategy_str,
                seed=(hash(f"{batch_id}_{seed}_{obj}") & 0x7FFFFFFF),
            )
            for i in range(n_samples):
                row = {
                    "seed": seed,
                    "objective": obj,
                    "batch_id": batch_id,
                }
                for pname in param_ranges:
                    row[pname] = samples_dict[pname][i]
                rows.append(row)

    return pd.DataFrame(rows)


def append_samples_to_csv(
    n: int,
    active_params: Dict[str, Dict[str, List[float]]],
    csv_path: str,
    strategy: str = "sobol",
    seed: Optional[int] = None,
    **kwargs,
) -> None:
    """Generate samples and append to an existing CSV file.

    If the file does not exist, it will be created with a header.

    Args:
        n: Number of samples.
        active_params: Dict mapping param name -> {'range': [low, high]}.
        csv_path: Path to the CSV file.
        strategy: Sampling strategy.
        seed: Random seed.
        **kwargs: Additional arguments passed to the sampler.
    """
    import pandas as pd

    df = generate_samples_dataframe(n, active_params, strategy, seed, **kwargs)
    header = not pd.io.common.file_exists(csv_path) if hasattr(pd.io.common, 'file_exists') else True
    try:
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
        header = False
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    df.to_csv(csv_path, index=False)