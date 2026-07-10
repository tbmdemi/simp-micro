"""
Adaptive sampling orchestrator: decides what to do next based on coverage.

Workflow:
  1. After each batch, run coverage analysis on accumulated results.
  2. Based on coverage gaps, objective improvement, and convergence,
     decide: stop / refine locally / expand globally.
  3. Generate the next BatchConfig accordingly.

Decision logic:
  - If objective has not improved for 2+ batches AND coverage is adequate → STOP.
  - If sparsity > 30% of property space → expand: add new seeds/objectives.
  - If sparsity < 10% but best objective is still far from theoretical →
    refine: narrow param bounds around best samples.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.multi_batch.coverage import coverage_report
from pipeline.multi_batch.params import BatchConfig, BatchMode, SamplingStrategy


def _load_accumulated_results(summaries: List[Dict]) -> List[Dict]:
    """Load all individual results from batch summary output dirs.

    Each summary has 'output_dir' pointing to where batch_{id}_results.json lives.

    Args:
        summaries: List of batch summary dicts.

    Returns:
        Combined list of result dicts.
    """
    import json
    import os

    all_results = []
    for s in summaries:
        batch_id = s.get('batch_id', 'unknown')
        out_dir = s.get('output_dir')
        if not out_dir or not os.path.isdir(out_dir):
            continue
        json_path = os.path.join(out_dir, f'batch_{batch_id}_results.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                payload = json.load(f)
            all_results.extend(payload.get('results', []))
    return all_results


def decide_next_action(
    summaries: List[Dict],
    config: 'PipelineConfig',  # noqa: F821 — forward ref at module level fine
    property_dims: Tuple[str, ...] = ('v12', 'v21', 'obj_value'),
    max_batches: int = 5,
    improvement_patience: int = 2,
    sparsity_expand_threshold: float = 0.30,
    sparsity_refine_threshold: float = 0.10,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """Analyze accumulated results and decide next batch action.

    Args:
        summaries: Batch summaries so far.
        config: The PipelineConfig (to read current active params, etc.).
        property_dims: Dimensions for coverage analysis.
        max_batches: Halt if we've hit this many batches.
        improvement_patience: Halt if no improvement for N batches.
        sparsity_expand_threshold: Fraction of sparse bins above which = expand.
        sparsity_refine_threshold: Fraction below which = refine.

    Returns:
        Dict with keys:
            'action': 'stop' | 'continue_sampling' | 'refine' | 'expand'
            'reason': str explanation
            'next_config': BatchConfig | None (if action != 'stop')
            'coverage': coverage_report output
    """
    n_completed = len(summaries)

    # ── 1. Load results ──
    all_results = _load_accumulated_results(summaries)

    # ── 2. Coverage analysis ──
    cov = coverage_report(all_results, dims=property_dims)

    n_sparse = cov.get('sparsity', {}).get('n_sparse_regions', 0)
    total_bins = max(1, _estimate_total_bins(all_results, property_dims))
    sparsity_frac = n_sparse / total_bins if total_bins > 0 else 1.0

    # ── 3. Objective trend ──
    best_objs = []
    for s in summaries:
        best_v = s.get('best_per_combo', {})
        vals = [v.get('obj_value', float('inf')) for v in best_v.values()
                if v.get('obj_value') is not None]
        if vals:
            best_objs.append(min(vals))

    n_no_improvement = 0
    if len(best_objs) >= 2:
        for i in range(len(best_objs) - 1, 0, -1):
            if best_objs[i] >= best_objs[i - 1] * 0.995:  # <0.5% improvement
                n_no_improvement += 1
            else:
                break

    # ── 4. Decision logic ──
    decision: Dict = {
        'n_batches_completed': n_completed,
        'n_valid_results': len(all_results),
        'best_objective_progress': {
            'values': best_objs,
            'n_batches_no_improvement': n_no_improvement,
        },
        'coverage': {
            'n_sparse_regions': n_sparse,
            'total_bins': total_bins,
            'sparsity_fraction': round(sparsity_frac, 4),
        },
        'action': None,
        'reason': '',
        'next_config': None,
    }

    # 4a. Stop conditions
    if n_completed >= max_batches:
        decision['action'] = 'stop'
        decision['reason'] = (f'Reached max batches ({max_batches}). '
                              f'Sparsity={sparsity_frac:.1%}.')
        return decision

    if n_no_improvement >= improvement_patience and sparsity_frac < sparsity_refine_threshold:
        decision['action'] = 'stop'
        decision['reason'] = (f'Objective stagnant for {n_no_improvement} batches '
                              f'and coverage adequate (sparsity={sparsity_frac:.1%}).')
        return decision

    # 4b. Expand: high sparsity -> add more seeds/objectives or wider ranges
    if sparsity_frac > sparsity_expand_threshold:
        decision['action'] = 'expand'
        decision['reason'] = (f'High sparsity ({sparsity_frac:.1%}). '
                              f'Adding more exploratory samples.')

        # Enrich strategy: wider bounds or more seeds
        next_id = n_completed + 1
        expanded_seeds = list(config.get('active_seeds', [])) if isinstance(config, dict) else (list(config.active_seeds) if hasattr(config, 'active_seeds') else [])
        # If few seeds, add more
        if len(expanded_seeds) < 5:
            all_seeds = [
                'circle', 'square', 'hourglass', 'four_circle', 'hexagonal',
                'nine_circle', 'cross_rectangular', 'grid_circular_voids',
                'small_square_cross', 'circle_half_quarter',
            ]
            for s in all_seeds:
                if s not in expanded_seeds:
                    expanded_seeds.append(s)
                    break  # Add one new seed per expand cycle

        expanded_objectives = list(config.get('active_objectives', [])) if isinstance(config, dict) else (list(config.active_objectives) if hasattr(config, 'active_objectives') else [])
        if len(expanded_objectives) < 3:
            for o in ['auxetic', 'first', 'second']:
                if o not in expanded_objectives:
                    expanded_objectives.append(o)
                    break

        # Wider param range: expand 20% beyond current
        current_active = config.get('active', {}) if isinstance(config, dict) else config.active
        expanded_active = deepcopy(current_active)
        for pname, prange in expanded_active.items():
            pmin, pmax = prange['range']
            span = pmax - pmin
            prange['range'] = [pmin - 0.2 * span, pmax + 0.2 * span]

        current_batches = config.get('batches', []) if isinstance(config, dict) else config.batches
        ref_n = current_batches[0].n_samples if current_batches else 120

        decision['next_config'] = BatchConfig(
            batch_id=next_id,
            n_samples=max(100, int(ref_n * 1.5)),
            strategy=SamplingStrategy.SOBOL,
            mode=BatchMode.EXPLORE,
            objectives=expanded_objectives,
            seeds=expanded_seeds,
        )
        return decision

    # 4c. Refine: low sparsity, still improving -> narrow down
    decision['action'] = 'refine'
    decision['reason'] = f'Coverage adequate, objective still improving. Refining.'

    next_id = n_completed + 1
    # Narrow parameter ranges around the best samples
    current_active = config.get('active', {}) if isinstance(config, dict) else config.active
    narrowed_active = _narrow_params(all_results, current_active)
    ref_objectives = config.get('active_objectives', ['auxetic']) if isinstance(config, dict) else (config.active_objectives if hasattr(config, 'active_objectives') else ['auxetic'])
    ref_seeds = config.get('active_seeds', ['circle']) if isinstance(config, dict) else (config.active_seeds if hasattr(config, 'active_seeds') else ['circle'])

    decision['next_config'] = BatchConfig(
        batch_id=next_id,
        n_samples=80,  # Fewer samples for refinement
        strategy=SamplingStrategy.OPTIMIZED_LHS,
        mode=BatchMode.REFINE,
        objectives=ref_objectives,
        seeds=ref_seeds,
    )
    # Attach narrowed ranges
    decision['next_config'].narrowed_params = narrowed_active

    return decision


def _estimate_total_bins(
    results: List[Dict],
    dims: Tuple[str, ...],
) -> int:
    """Estimate number of bins used in sparsity analysis."""
    import numpy as np

    valid = [r for r in results if r.get('success')]
    n = len(valid)
    if n < 10:
        return 1
    n_bins = max(8, int(np.cbrt(n)))
    bins_per_dim = max(3, int(n_bins ** (1.0 / len(dims))))
    return bins_per_dim ** len(dims)


def _narrow_params(
    all_results: List[Dict],
    current_active: Dict[str, Dict],
    quantile: float = 0.2,
) -> Dict:
    """Narrow active parameter ranges around best-performing samples.

    Uses the top quantile of samples (by objective) to define new bounds.

    Args:
        all_results: Accumulated results.
        current_active: Current active parameter definitions.
        quantile: Fraction of best samples to use (0.2 = top 20%).

    Returns:
        Dict with same structure as current_active but narrower ranges.
    """
    valid = [r for r in all_results
             if r.get('success') and r.get('obj_value') is not None]
    if len(valid) < 10:
        return deepcopy(current_active)

    # Sort by objective (lower = better)
    sorted_results = sorted(valid, key=lambda x: x['obj_value'])
    n_best = max(5, int(len(sorted_results) * quantile))
    best = sorted_results[:n_best]

    narrowed = {}
    for pname, pdef in current_active.items():
        pmin, pmax = pdef['range']
        vals = [r.get('params', {}).get(pname) for r in best]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if len(vals) < 3:
            narrowed[pname] = deepcopy(pdef)
            continue

        new_min = max(pmin, float(np.percentile(vals, 15)))
        new_max = min(pmax, float(np.percentile(vals, 85)))
        # Ensure at least 10% of original range
        min_span = 0.1 * (pmax - pmin)
        if new_max - new_min < min_span:
            center = (new_min + new_max) / 2
            new_min = max(pmin, center - min_span / 2)
            new_max = min(pmax, center + min_span / 2)

        narrowed[pname] = {
            'range': [new_min, new_max],
        }

    return narrowed