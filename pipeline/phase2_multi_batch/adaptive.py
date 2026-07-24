"""
Adaptive sampling orchestrator: after each batch, runs coverage analysis on
accumulated results and decides stop / refine / expand, producing the next
BatchConfig.

Decision logic:
  - Objective stagnant for 2+ batches AND coverage adequate -> STOP.
  - Sparsity > 30% of property space -> EXPAND (new seeds/objectives).
  - Sparsity < 10% but objective still improving -> REFINE (narrow bounds
    around best samples).

Manufacturability (roadmap 6.2/6.3, xem runner.py::evaluate_single):
    Phân tích ngược 7.920 mẫu Phase 2 đã có (EXPERIMENT_LOG.md mục "Phase 2
    — Manufacturability") cho thấy tham số DOE liên tục (volfrac, penal,
    rmin, move, void_size_frac) hầu như KHÔNG tương quan với
    manufacturability (|Spearman r|<0,12 mọi trường hợp, kể cả khi tách
    riêng từng seed) - trong khi SEED giải thích chênh lệch tới 8 lần
    (7,9% ở hexagonal tới 62,8% ở reentrant_bowtie). Vì vậy _narrow_params()
    (vốn chỉ narrow tham số liên tục) không phải đòn bẩy đúng cho
    manufacturability - đòn bẩy đúng là TRỌNG SỐ SEED, thêm ở
    compute_seed_sample_allocation() bên dưới. _narrow_params() vẫn được
    sửa nhẹ (ưu tiên mẫu manufacturable khi chọn "best" để narrow) để không
    vô tình lệch tham số theo hướng ngược lại, nhưng đây không phải sửa
    chính.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.phase2_multi_batch.coverage import (
    coverage_report,
    find_sparse_regions,
    recommend_new_samples,
    seed_manufacturability_report,
)
from pipeline.phase2_multi_batch.params import BatchConfig, BatchMode, SamplingStrategy


# Fix H1: reentrant_bowtie was missing from this list, so _fill_seeds() could
# never add it back in when expanding seeds.
_ALL_SEEDS = [
    'circle', 'square', 'hourglass', 'four_circle', 'hexagonal',
    'nine_circle', 'cross_rectangular', 'grid_circular_voids',
    'small_square_cross', 'circle_half_quarter', 'reentrant_bowtie',
]


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


# Fix L2: denominator takes abs(curr) too (not just abs(prev)), so a near-zero
# prev no longer produces a spuriously huge relative improvement.
def _relative_improvement(curr: float, prev: float) -> float:
    """Compute relative improvement when minimizing (lower = better).

    Positive return value means the objective improved. Fractional, relative
    to max(|prev|, |curr|, 1e-10) — see Fix L2 above.

    Args:
        curr: Current best objective value.
        prev: Previous best objective value.

    Returns:
        Relative improvement fraction (positive = curr is better).
    """
    denom = max(abs(prev), abs(curr), 1e-10)
    # Positive when curr < prev (minimization, lower is better)
    return (prev - curr) / denom


def _fill_seeds(expanded_seeds: List[str],
                 seed_scores: Optional[Dict[str, float]] = None) -> List[str]:
    """Add seeds from _ALL_SEEDS until we have at least 5.

    seed_scores: nếu truyền vào (xem seed_manufacturability_report/
    compute_seed_sample_allocation), ưu tiên thêm seed điểm cao hơn trước
    thay vì theo thứ tự cố định trong _ALL_SEEDS. None (mặc định) giữ đúng
    hành vi gốc (thứ tự cố định) - tương thích ngược 100%.
    """
    result = list(expanded_seeds)
    candidates = [s for s in _ALL_SEEDS if s not in result]
    if seed_scores:
        candidates = sorted(candidates, key=lambda s: seed_scores.get(s, 0.0), reverse=True)
    for s in candidates:
        if len(result) >= 5:
            break
        result.append(s)
    return result


def compute_seed_sample_allocation(
    all_results: List[Dict],
    seeds: List[str],
    n_samples_baseline: int,
    min_weight_frac: float = 0.3,
    min_n_for_score: int = 20,
) -> Dict[str, int]:
    """Phân bổ LẠI ngân sách mẫu giữa các seed, LỆCH theo hiệu năng thay vì
    chia đều 1/n_seeds (hành vi gốc trước đây - xem sampling.py::
    generate_design, luôn cho mỗi seed đúng n_samples mẫu). Tổng ngân sách
    được BẢO TOÀN (== n_samples_baseline * len(seeds), giống hệt tổng số
    mẫu hành vi gốc sẽ tạo ra), chỉ phân bổ khác nhau giữa các seed - KHÔNG
    làm giảm tổng kích thước batch.

    n_samples_baseline: số mẫu MỖI seed sẽ nhận nếu chia đều (đúng ý nghĩa
    BatchConfig.n_samples gốc truyền vào generate_design()).

    Điểm mỗi seed = joint_rate (đồng thời auxetic VÀ manufacturable, xem
    seed_manufacturability_report) nếu đã có đủ dữ liệu (>= min_n_for_score
    mẫu VÀ có dữ liệu manufacturability), fallback về auxetic_rate nếu chưa
    có dữ liệu manufacturability, fallback về điểm trung lập (trung bình
    các seed đã biết) nếu seed hoàn toàn mới/quá ít dữ liệu - để không phạt
    oan 1 seed mới thêm vào.

    min_weight_frac: sàn trọng số tối thiểu (tỷ lệ so với phần chia đều
    1/n_seeds) - đảm bảo KHÔNG seed nào bị loại hoàn toàn (mỗi seed vẫn góp
    phần đa dạng hình học cho dataset ML dù joint_rate đo được ở 1 batch
    nhỏ tình cờ bằng 0 - tránh phạt nặng vì nhiễu thống kê ở mẫu nhỏ).

    Returns:
        Dict seed -> n_samples (>= 1 mỗi seed), tổng xấp xỉ
        n_samples_baseline * len(seeds) (có thể lệch vài đơn vị do làm tròn).
    """
    seed_report = seed_manufacturability_report(all_results)

    raw_scores: Dict[str, Optional[float]] = {}
    for s in seeds:
        info = seed_report.get(s)
        if info is None or info['n'] < min_n_for_score:
            raw_scores[s] = None
            continue
        raw_scores[s] = (
            info['joint_rate'] if info['joint_rate'] is not None else info['auxetic_rate']
        )

    known = [v for v in raw_scores.values() if v is not None]
    neutral = float(np.mean(known)) if known else 0.2  # 0.2: giả định trung lập khi chưa có gì

    scores = {s: (v if v is not None else neutral) for s, v in raw_scores.items()}

    uniform_share = sum(scores.values()) / len(scores) if scores else 0.0
    floor = min_weight_frac * uniform_share
    adjusted = {s: max(v, floor) for s, v in scores.items()}

    total_budget = n_samples_baseline * len(seeds)
    total_score = sum(adjusted.values())
    if total_score <= 0:
        # tất cả điểm bằng 0 (không nên xảy ra vì floor > 0 khi uniform_share > 0,
        # nhưng phòng trường hợp toàn bộ scores=0.0 và floor cũng =0) -> chia đều.
        return {s: n_samples_baseline for s in seeds}

    allocation = {}
    for s in seeds:
        frac = adjusted[s] / total_score
        allocation[s] = max(1, round(frac * total_budget))

    return allocation


def decide_next_action(
    summaries: List[Dict],
    config: 'PipelineConfig',  # noqa: F821 — forward ref
    property_dims: Tuple[str, ...] = ('v12', 'v21', 'obj_value'),
    max_batches: int = 5,
    improvement_patience: int = 2,
    sparsity_expand_threshold: float = 0.30,
    sparsity_refine_threshold: float = 0.10,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_sparse_targeted: int = 20,
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
        param_ranges: Current active param ranges (name -> (lo, hi)).
        n_sparse_targeted: How many targeted samples for sparse regions in refine.

    Returns:
        Dict with keys:
            'action': 'stop' | 'continue_sampling' | 'refine' | 'expand'
            'reason': str explanation
            'next_config': BatchConfig | None (if action != 'stop')
            'coverage': coverage_report output
            'sparse_targeted_params': List[Dict] of targeted param suggestions
    """
    # ── Fix M1: param_ranges=None early guard ──
    if param_ranges is not None and not isinstance(param_ranges, dict):
        raise ValueError("param_ranges must be a dict or None")

    n_completed = len(summaries)

    # ── 1. Load results ──
    all_results = _load_accumulated_results(summaries)

    # ── 1b. Seed-level manufacturability (roadmap 6.2/6.3 - xem docstring
    # đầu file: đây là đòn bẩy chính cho manufacturability, không phải
    # narrow tham số liên tục) ──
    seed_report = seed_manufacturability_report(all_results)
    seed_scores = {
        s: (info['joint_rate'] if info['joint_rate'] is not None else info['auxetic_rate'])
        for s, info in seed_report.items()
    }

    # ── 2. Coverage analysis ──
    cov = coverage_report(all_results, dims=property_dims)

    n_sparse = cov.get('sparsity', {}).get('n_sparse_regions', 0)
    total_bins = max(1, _estimate_total_bins(all_results, property_dims))
    sparsity_frac = n_sparse / total_bins if total_bins > 0 else 1.0

    # Get detailed sparse region info for targeted sampling
    sparse_regions = cov.get('sparsity', {}).get('regions', [])

    # ── 3. Objective trend ──
    best_objs = []
    for s in summaries:
        best_v = s.get('best_per_combo', {})
        vals = [v.get('v12', float('inf')) for v in best_v.values()
                if v.get('v12') is not None]
        if vals:
            best_objs.append(min(vals))

    # Count consecutive batches WITHOUT meaningful improvement.
    # Walks backward through history: each pair where rel_imp < 0.5% counts.
    n_no_improvement = 0
    if len(best_objs) >= 2:
        for i in range(len(best_objs) - 1, 0, -1):
            prev = best_objs[i - 1]
            curr = best_objs[i]
            rel_imp = _relative_improvement(curr, prev)
            # Less than 0.5% relative improvement = stagnant
            if rel_imp < 0.005:
                n_no_improvement += 1
            else:
                break

    # Also compute the *overall* improvement from first to latest batch
    overall_improving = False
    if len(best_objs) >= 2:
        first_val = best_objs[0]
        last_val = best_objs[-1]
        overall_imp = _relative_improvement(last_val, first_val)
        overall_improving = overall_imp > 0.01  # >1% overall improvement

    # Compute pairwise improvements for logging
    pairwise_improvements = []
    for i in range(1, len(best_objs)):
        pairwise_improvements.append(
            _relative_improvement(best_objs[i], best_objs[i - 1])
        )

    # ── 4. Decision logic ──
    decision: Dict = {
        'n_batches_completed': n_completed,
        'n_valid_results': len(all_results),
        'best_objective_progress': {
            'values': best_objs,
            'relative_improvements': pairwise_improvements,
            'n_batches_no_improvement': n_no_improvement,
            'overall_improving': overall_improving,
        },
        'coverage': {
            'n_sparse_regions': n_sparse,
            'total_bins': total_bins,
            'sparsity_fraction': round(sparsity_frac, 4),
        },
        'action': None,
        'reason': '',
        'next_config': None,
        'sparse_targeted_params': [],
        'seed_manufacturability': seed_report,
    }

    # 4a. Stop conditions — batch limit reached
    if n_completed >= max_batches:
        decision['action'] = 'stop'
        decision['reason'] = (f'Reached max batches ({max_batches}). '
                              f'Sparsity={sparsity_frac:.1%}.')
        return decision

    # 4b. Check if previous batch was refinement and it failed to improve
    has_prior_refinement = any(
        s.get('mode') == 'refine' or s.get('mode') == BatchMode.REFINE.value
        for s in summaries
    ) if len(summaries) >= 2 else False
    refinement_failed = False
    refine_imp = 0.0
    if has_prior_refinement and len(best_objs) >= 2:
        last_two = best_objs[-2:]
        worst_prev_after_refine = max(last_two)
        best_current = min(last_two)
        refine_imp = _relative_improvement(best_current, worst_prev_after_refine)
        if refine_imp < 0.005:
            refinement_failed = True

    # 4c. Stop: objective stagnant AND coverage adequate
    if n_no_improvement >= improvement_patience and sparsity_frac < sparsity_refine_threshold:
        decision['action'] = 'stop'
        decision['reason'] = (f'Objective stagnant for {n_no_improvement} batch(es) '
                              f'and coverage adequate (sparsity={sparsity_frac:.1%}). '
                              f'Best values: {[round(v, 4) for v in best_objs]}.')
        return decision

    # 4d. Stop: refinement round(s) produced no objective gain
    if refinement_failed and n_no_improvement >= 1:
        decision['action'] = 'stop'
        decision['reason'] = (f'Refinement round did not improve objective '
                              f'(rel. change={refine_imp:.3%}) after '
                              f'{n_no_improvement} stagnant batch(es). '
                              f'Coverage sparsity={sparsity_frac:.1%}. '
                              f'Best obj history: {[round(v, 4) for v in best_objs]}.')
        return decision

    # 4e. Stop: no improvement recently AND objective is NOT improving overall
    #     (catches the case where coverage is moderate but we're not progressing)
    if n_no_improvement >= 1 and not overall_improving and sparsity_frac < 0.20:
        decision['action'] = 'stop'
        decision['reason'] = (f'Objective not improving overall '
                              f'({n_no_improvement} stagnant batch(es), '
                              f'overall_improving={overall_improving}). '
                              f'Best obj history: {[round(v, 4) for v in best_objs]}.')
        return decision

    # 4f. Early exploration: fewer than 2 batches → force expand
    if n_completed < 2:
        decision['action'] = 'expand'
        decision['reason'] = (f'Only {n_completed} batch(es) completed. '
                              f'Need more exploration before refinement.')

        next_id = n_completed + 1
        expanded_seeds = list(config.get('active_seeds', [])) if isinstance(config, dict) else (list(config.active_seeds) if hasattr(config, 'active_seeds') else [])
        expanded_seeds = _fill_seeds(expanded_seeds, seed_scores=seed_scores)

        expanded_objectives = list(config.get('active_objectives', [])) if isinstance(config, dict) else (list(config.active_objectives) if hasattr(config, 'active_objectives') else [])
        if len(expanded_objectives) < 3:
            for o in ['auxetic']:
                if o not in expanded_objectives:
                    expanded_objectives.append(o)
                    break

        decision['next_config'] = BatchConfig(
            batch_id=next_id,
            n_samples=120,
            strategy=SamplingStrategy.SOBOL,
            mode=BatchMode.EXPLORE,
            objectives=expanded_objectives,
            seeds=expanded_seeds,
            param_ranges=param_ranges,
        )
        return decision

    # 4g. Expand: high sparsity — add more seeds/objectives or wider ranges
    if sparsity_frac > sparsity_expand_threshold:
        decision['action'] = 'expand'
        decision['reason'] = (f'High sparsity ({sparsity_frac:.1%}). '
                              f'Adding more exploratory samples.')

        next_id = n_completed + 1
        expanded_seeds = list(config.get('active_seeds', [])) if isinstance(config, dict) else (list(config.active_seeds) if hasattr(config, 'active_seeds') else [])
        expanded_seeds = _fill_seeds(expanded_seeds, seed_scores=seed_scores)

        expanded_objectives = list(config.get('active_objectives', [])) if isinstance(config, dict) else (list(config.active_objectives) if hasattr(config, 'active_objectives') else [])
        if len(expanded_objectives) < 3:
            for o in ['auxetic']:
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
            param_ranges=param_ranges,
        )
        return decision

    # ── 4h. Refine + optionally fill sparse regions ──
    decision['action'] = 'refine'
    next_id = n_completed + 1
    current_active = config.get('active', {}) if isinstance(config, dict) else config.active
    # Pass n_completed to control aggressiveness (Issue #1 fix)
    narrowed_active = _narrow_params(
        all_results, current_active,
        n_batches_completed=n_completed,
    )

    # Convert narrowed_active to param_ranges format {name: (lo, hi)}
    narrowed_param_ranges: Dict[str, Tuple[float, float]] = {}
    for pname, pdef in narrowed_active.items():
        lo, hi = pdef['range']
        narrowed_param_ranges[pname] = (lo, hi)

    # Merge with any original params not in narrowed
    if param_ranges:
        for pname, prange in param_ranges.items():
            if pname not in narrowed_param_ranges:
                narrowed_param_ranges[pname] = prange

    # Check if narrowing actually reduced any range
    did_narrow = False
    if param_ranges:
        for pname, (nlo, nhi) in narrowed_param_ranges.items():
            if pname in param_ranges:
                olo, ohi = param_ranges[pname]
                orig_span = ohi - olo
                new_span = nhi - nlo
                if new_span < 0.95 * orig_span:
                    did_narrow = True
                    break

    if did_narrow:
        n_narrowed = sum(
            1 for p in narrowed_param_ranges
            if p in param_ranges
            and (
                narrowed_param_ranges[p][1] - narrowed_param_ranges[p][0]
            ) < 0.95 * (
                param_ranges[p][1] - param_ranges[p][0]
            )
        )
        reason = (
            f'Coverage adequate, objective improving. '
            f'Refining: narrowed {n_narrowed} parameter range(s).'
        )
    else:
        n_original = len(param_ranges) if param_ranges else 0
        reason = (f'Coverage adequate, narrowing insufficient '
                  f'({len(narrowed_param_ranges)}/{n_original} params narrowed). '
                  f'Maintaining current bounds + sparse targeting.')

    # If sparse regions exist, generate targeted recommendations
    sparse_targeted_params = []
    if sparse_regions and param_ranges:
        targeted = recommend_new_samples(
            all_results=all_results,
            sparse_regions=sparse_regions[:3],
            param_space={k: v for k, v in param_ranges.items()},
            n_recommend=n_sparse_targeted,
            dims=property_dims[:2],
        )
        sparse_targeted_params = targeted
        if targeted:
            reason += (f' Also targeting {len(targeted)} samples at '
                       f'{min(len(sparse_regions), 3)} sparse region(s).')

    decision['reason'] = reason

    # Determine total sample count
    base_n = 80
    if sparse_targeted_params:
        total_n = base_n + len(sparse_targeted_params)
    else:
        total_n = base_n

    ref_objectives = config.get('active_objectives', ['auxetic']) if isinstance(config, dict) else (
        config.active_objectives if hasattr(config, 'active_objectives') else ['auxetic']
    )
    ref_seeds = config.get('active_seeds', ['circle']) if isinstance(config, dict) else (
        config.active_seeds if hasattr(config, 'active_seeds') else ['circle']
    )

    # roadmap 6.2/6.3: phân bổ mẫu LỆCH theo seed thay vì chia đều 1/n_seeds
    # (xem compute_seed_sample_allocation docstring + module docstring đầu
    # file) - đây là đòn bẩy chính cho manufacturability, refine mode là
    # nơi hợp lý nhất để áp dụng (mode "tập trung vào vùng đã biết là tốt").
    n_samples_per_seed = compute_seed_sample_allocation(
        all_results, ref_seeds, total_n,
    )

    decision['next_config'] = BatchConfig(
        batch_id=next_id,
        n_samples=total_n,
        strategy=SamplingStrategy.OPTIMIZED_LHS,
        mode=BatchMode.REFINE,
        objectives=ref_objectives,
        seeds=ref_seeds,
        param_ranges=narrowed_param_ranges,  # Use narrowed ranges, not original!
        n_samples_per_seed=n_samples_per_seed,
    )
    # Attach metadata for the caller
    decision['next_config'].narrowed_params = narrowed_active
    decision['sparse_targeted_params'] = sparse_targeted_params

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
    quantile: Optional[float] = None,
    n_batches_completed: int = 1,
) -> Dict:
    """Narrow active parameter ranges around best-performing samples.

    Uses the top quantile of samples (by objective) to define new bounds.
    The narrowing becomes progressively more aggressive with more batches:
    - Batch 2: top 20% → 15th–85th percentile (gentle)
    - Batch 3: top 15% → 10th–80th percentile (moderate)
    - Batch 4+: top 10% → 5th–75th percentile (aggressive)

    Args:
        all_results: Accumulated results.
        current_active: Current active parameter definitions.
        quantile: Override fraction of best samples to use. If None, auto-calculated.
        n_batches_completed: Number of batches already completed (determines aggressiveness).

    Returns:
        Dict with same structure as current_active but narrower ranges.
    """
    valid = [r for r in all_results
         if r.get('success') and r.get('v12') is not None]
    if len(valid) < 10:
        return deepcopy(current_active)

    # Auto-calculate quantile based on batch count:
    #   Batch 2: 0.20 (gentle)
    #   Batch 3: 0.15 (moderate)
    #   Batch 4+: 0.10 (aggressive)
    if quantile is None:
        if n_batches_completed >= 4:
            quantile = 0.10
            lo_pct, hi_pct = 5, 75
        elif n_batches_completed >= 3:
            quantile = 0.15
            lo_pct, hi_pct = 10, 80
        else:
            quantile = 0.20
            lo_pct, hi_pct = 15, 85
    else:
        # Default percentiles for manual quantile
        lo_pct, hi_pct = 15, 85

    # Sort by objective (lower = better), ưu tiên mẫu manufacturable trước
    # (roadmap 6.2/6.3) - chỉ phạt khi passes_all TƯỜNG MINH là False; None
    # (kết quả cũ/mock chưa có instrumentation này) coi như trung lập, giữ
    # đúng hành vi gốc (sort thuần theo v12) khi không có dữ liệu manuf.
    sorted_results = sorted(
        valid, key=lambda x: (x.get('passes_all') is False, x['v12'])
    )
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

        new_min = max(pmin, float(np.percentile(vals, lo_pct)))
        new_max = min(pmax, float(np.percentile(vals, hi_pct)))
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