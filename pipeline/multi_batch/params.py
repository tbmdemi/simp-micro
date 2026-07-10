"""
Cấu hình tham số cho multi-batch adaptive sampling pipeline.

Lưu trữ:
  - Fixed parameters (rmin, move, penal)
  - Active parameter ranges (volfrac, void_size_frac)
  - Cấu hình từng batch (strategy, seed, sample size)
  - Đường dẫn output
"""

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


# ── Enums ──

class SamplingStrategy(str, Enum):
    """Available sampling strategies."""
    SOBOL = 'sobol'
    LHS = 'lhs'
    OPTIMIZED_LHS = 'optimized_lhs'
    RANDOM = 'random'


class BatchMode(str, Enum):
    """Batch sampling mode."""
    EXPLORE = 'explore'        # space-filling, cover the domain
    REFINE = 'refine'          # focus on promising regions
    TARGETED = 'targeted'      # directed sampling based on gradients / surrogate
    VALIDATE = 'validate'      # validation of specific points


# ── Defaults ──

FIXED_PARAMETERS: Dict[str, float] = {
    "rmin": 1.5,
    "move": 0.2,
    "penal": 3.0,
}

ACTIVE_PARAMETERS: Dict[str, Dict[str, List[float]]] = {
    "volfrac": {"range": [0.3, 0.7]},
    "void_size_frac": {"range": [0.1, 0.4]},
}


# ── Dataclasses ──

@dataclass
class BatchConfig:
    """Configuration for a single batch."""

    batch_id: int
    """Batch number (1, 2, 3, ...)."""

    n_samples: int = 120
    """Number of samples in this batch."""

    strategy: SamplingStrategy = SamplingStrategy.SOBOL
    """Sampling strategy."""

    mode: BatchMode = BatchMode.EXPLORE
    """Batch mode (explore / refine / targeted / validate)."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    objectives: List[str] = field(default_factory=lambda: ["first", "second", "auxetic"])
    """Objective functions to run (list of 'first', 'second', 'auxetic')."""

    seeds: List[str] = field(
        default_factory=lambda: [
            "circle",
            "circle_half_quarter",
            "cross_rectangular",
            "four_circle",
            "grid_circular_voids",
            "hexagonal",
            "hourglass",
            "nine_circle",
            "small_square_cross",
            "square",
        ]
    )
    """Seed shape names to use."""

    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    """Parameter ranges for this batch (name -> (low, high)). Overrides global ranges if set."""

    output_dir: str = ""
    """Output directory override. If empty, uses default."""

    def get_output_dir(self, base_dir: str = "outputs/pipeline") -> str:
        """Get the output directory for this batch.

        Args:
            base_dir: Base output directory.

        Returns:
            Path string like 'outputs/pipeline/phase2' for batch 2.
        """
        if self.output_dir:
            return self.output_dir
        return os.path.join(base_dir, f"phase{self.batch_id + 1}")


@dataclass
class PipelineConfig:
    """Complete multi-batch pipeline configuration."""

    fixed: Dict[str, float] = field(default_factory=lambda: dict(FIXED_PARAMETERS))
    """Fixed SIMP parameters."""

    active: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: dict(ACTIVE_PARAMETERS)
    )
    """Active parameter ranges."""

    active_seeds: List[str] = field(default_factory=list)
    """Seed shapes to include in the study."""

    active_objectives: List[str] = field(default_factory=list)
    """Objectives to include."""

    batches: List[BatchConfig] = field(default_factory=list)
    """List of batch configurations."""

    base_output_dir: str = "outputs/pipeline"
    """Base directory for all outputs."""

    def add_batch(self, batch: BatchConfig) -> None:
        """Add a batch configuration.

        Args:
            batch: BatchConfig instance.
        """
        self.batches.append(batch)

    def save(self, path: str) -> None:
        """Save pipeline configuration to JSON.

        Args:
            path: Output JSON file path.
        """
        def _serialize(o: Any) -> Any:
            if isinstance(o, Enum):
                return o.value
            if hasattr(o, '__dict__'):
                return asdict(o)  # type: ignore
            return o

        data = {
            "fixed": dict(self.fixed),
            "active": {k: dict(v) for k, v in self.active.items()},
            "batches": [_serialize(b) for b in self.batches],
            "base_output_dir": self.base_output_dir,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load pipeline configuration from JSON.

        Args:
            path: Input JSON file path.

        Returns:
            PipelineConfig instance.
        """
        with open(path) as f:
            data = json.load(f)

        config = cls(
            fixed=data.get("fixed", dict(FIXED_PARAMETERS)),
            active=data.get("active", dict(ACTIVE_PARAMETERS)),
            base_output_dir=data.get("base_output_dir", "outputs/pipeline"),
        )

        for b_data in data.get("batches", []):
            # Convert strategy back to enum
            if "strategy" in b_data and isinstance(b_data["strategy"], str):
                b_data["strategy"] = SamplingStrategy(b_data["strategy"])
            # Convert mode back to enum
            if "mode" in b_data and isinstance(b_data["mode"], str):
                b_data["mode"] = BatchMode(b_data["mode"])
            config.add_batch(BatchConfig(**b_data))

        return config


# ── Helper functions ──

def load_phase1_params(summaries: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """Extract parameter ranges from phase 1 summary data.

    Args:
        summaries: List of phase 1 summary dicts, each containing
            a 'parameters' key with param_name -> {'range': [low, high]}.

    Returns:
        Dict mapping parameter name -> (low, high) tuples.
    """
    param_ranges: Dict[str, Tuple[float, float]] = {}

    for s in summaries:
        params = s.get('parameters', {})
        if not isinstance(params, dict):
            continue
        for pname, pdef in params.items():
            if isinstance(pdef, dict) and 'range' in pdef:
                r = pdef['range']
                if len(r) == 2:
                    # Keep the widest range seen
                    if pname not in param_ranges:
                        param_ranges[pname] = (float(r[0]), float(r[1]))
                    else:
                        cur_low, cur_high = param_ranges[pname]
                        new_low = min(cur_low, float(r[0]))
                        new_high = max(cur_high, float(r[1]))
                        param_ranges[pname] = (new_low, new_high)

    return param_ranges


def prepare_output(base_dir: str, batch_id: int, suffix: str = "") -> str:
    """Create and return an output directory path for a batch.

    Args:
        base_dir: Base output directory.
        batch_id: Batch number.
        suffix: Optional subdirectory suffix.

    Returns:
        Full path to the output directory.
    """
    path = os.path.join(base_dir, f"batch_{batch_id}{suffix}")
    os.makedirs(path, exist_ok=True)
    return path


def load_refined_parameters(path: str = "outputs/pipeline/phase1/refined_parameters.json") -> PipelineConfig:
    """Load Phase 1 refined parameters and create a default pipeline config.

    Args:
        path: Path to refined_parameters.json.

    Returns:
        PipelineConfig with defaults filled in.
    """
    with open(path) as f:
        refined = json.load(f)

    config = PipelineConfig()

    # Apply fixed parameters
    if "fixed_parameters" in refined:
        config.fixed.update(refined["fixed_parameters"])

    # Apply active parameters
    if "active_parameters" in refined:
        config.active.update(refined["active_parameters"])

    # Add default batch configs
    config.add_batch(
        BatchConfig(
            batch_id=1,
            n_samples=120,
            strategy=SamplingStrategy.SOBOL,
            seed=42,
        )
    )
    config.add_batch(
        BatchConfig(
            batch_id=2,
            n_samples=80,
            strategy=SamplingStrategy.LHS,
            seed=43,
        )
    )

    return config


def default_config() -> PipelineConfig:
    """Create a default PipelineConfig with two batches.

    Returns:
        PipelineConfig instance ready for use.
    """
    config = PipelineConfig()
    config.add_batch(
        BatchConfig(
            batch_id=1, n_samples=120, strategy=SamplingStrategy.SOBOL, seed=42
        )
    )
    config.add_batch(
        BatchConfig(
            batch_id=2, n_samples=80, strategy=SamplingStrategy.LHS, seed=43
        )
    )
    return config