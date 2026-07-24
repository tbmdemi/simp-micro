"""
Coverage analysis for property space exploration.

Assesses how well a batch (or combined batches) covers the property space
(v12 × v21 × objective), identifies sparse regions, and recommends
where to sample next.

Key functions:
  - compute_density_estimate: 2D/3D KDE over property space
  - find_sparse_regions: locate low-density hyperrectangles
  - coverage_report: textual + dict summary of coverage quality
  - recommend_new_samples: produce parameter hints for next batch
  - seed_manufacturability_report: per-seed auxetic/manufacturable rates
    (roadmap 6.2/6.3, xem runner.py::evaluate_single - 'passes_all' được đo
    TẠI THỜI ĐIỂM SINH, miễn phí, không tốn FE thêm). Kết quả phân tích
    ngược 7.920 mẫu Phase 2 (xem EXPERIMENT_LOG.md mục "Phase 2 —
    Manufacturability") cho thấy SEED là biến giải thích chi phối
    manufacturability (7,9%-62,8% tuỳ seed), không phải tham số DOE liên
    tục (|r|<0,12 mọi trường hợp) - hàm này tồn tại để adaptive.py dùng
    SEED làm đòn bẩy chính, thay vì chỉ narrow tham số liên tục như trước.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde


def _to_array(results: List[Dict], field: str) -> np.ndarray:
    """Extract numeric field from successful results, drop NaN/None."""
    vals = []
    for r in results:
        v = r.get(field)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.array(vals, dtype=float)


def compute_density_estimate(
    results: List[Dict],
    dims: Tuple[str, ...] = ('v12', 'v21'),
    bw_method: Optional[str] = 'scott',
) -> Dict:
    """Compute KDE density over the property space.

    Args:
        results: List of result dicts (must have 'success'=True entries).
        dims: Property dimensions to use.
        bw_method: Bandwidth method for KDE (None=Scott).

    Returns:
        Dict with:
            'points': (N, D) array of valid points
            'density': (N,) array of log-density values at points
            'bounds': [(min, max), ...] per dimension
            'grid': dict of meshgrid arrays if ndim <= 3
    """
    # Stack valid points
    arrays = []
    for d in dims:
        vals = _to_array(results, d)
        if len(vals) == 0:
            raise ValueError(f"No valid data for dimension '{d}'")
        arrays.append(vals)

    points = np.column_stack(arrays)
    n_valid = points.shape[0]
    n_dims = points.shape[1]

    if n_valid < 5:
        return {
            'n_valid': n_valid,
            'error': f'Too few points ({n_valid}) for density estimation',
        }

    # Remove outliers (0.5-99.5 percentile)
    lower = np.percentile(points, 0.5, axis=0)
    upper = np.percentile(points, 99.5, axis=0)
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    points_clean = points[mask]

    if points_clean.shape[0] < 5:
        points_clean = points

    # KDE
    try:
        kde = gaussian_kde(points_clean.T, bw_method=bw_method)
        density = kde(points_clean.T)
        log_density = np.log(np.maximum(density, 1e-30))
    except Exception:
        # Fallback: uniform density
        log_density = np.zeros(points_clean.shape[0])

    result: Dict = {
        'n_valid': n_valid,
        'points': points_clean.tolist(),
        'density': density.tolist() if 'density' in dir() else None,
        'log_density': log_density.tolist(),
        'bounds': [(float(points_clean[:, i].min()), float(points_clean[:, i].max()))
                   for i in range(n_dims)],
        'dimensions': list(dims),
    }

    # Grid for 2D visualisation
    if n_dims <= 3:
        n_grid = 50
        grids = []
        for i in range(n_dims):
            lo, hi = result['bounds'][i]
            grids.append(np.linspace(lo, hi, n_grid))
        mesh = np.meshgrid(*grids, indexing='ij')
        result['grid'] = {
            'axes': [g.tolist() for g in grids],
            'arrays': [m.tolist() for m in mesh],
        }

    return result


def find_sparse_regions(
    results: List[Dict],
    dims: Tuple[str, ...] = ('v12', 'v21'),
    n_regions: int = 5,
    density_threshold: float = 0.1,
) -> List[Dict]:
    """Identify sparse regions in property space using adaptive binning.

    Args:
        results: List of result dicts.
        dims: Property dimensions.
        n_regions: Max number of sparse regions to return.
        density_threshold: Fraction of max density below which = sparse.

    Returns:
        List of dicts with 'bounds', 'n_points', 'volume', 'density'.
    """
    arrays = []
    for d in dims:
        vals = _to_array(results, d)
        if len(vals) == 0:
            return []
        arrays.append(vals)

    points = np.column_stack(arrays)
    n_dims = points.shape[1]
    n_total = points.shape[0]

    if n_total < 10:
        return [{'note': f'Too few points ({n_total}) for spatial analysis'}]

    # Compute KDE on a coarse grid
    n_bins = max(8, int(np.cbrt(n_total)))
    bins_per_dim = max(3, int(n_bins ** (1.0 / n_dims)))

    bounds = [(float(points[:, i].min()), float(points[:, i].max()))
              for i in range(n_dims)]
    edges = [np.linspace(b[0], b[1], bins_per_dim + 1) for b in bounds]

    # Count points per bin
    bin_counts = np.zeros([bins_per_dim] * n_dims)
    for p in points:
        idxs = []
        for i in range(n_dims):
            idx = np.searchsorted(edges[i][1:], p[i], side='left')
            idx = min(idx, bins_per_dim - 1)
            idxs.append(idx)
        bin_counts[tuple(idxs)] += 1

    max_count = bin_counts.max() if bin_counts.max() > 0 else 1

    # Find sparse bins (below threshold)
    sparse_mask = bin_counts < (density_threshold * max_count)
    sparse_indices = np.argwhere(sparse_mask)

    # Rank by sparsity and aggregate nearby
    sparsity_rank = []
    for idx in sparse_indices:
        count = bin_counts[tuple(idx)]
        vol = np.prod([(edges[i][idx[i]+1] - edges[i][idx[i]]) for i in range(n_dims)])
        sparsity_rank.append({
            'index': idx.tolist(),
            'n_points': int(count),
            'volume': float(vol),
            'bounds': [[float(edges[i][idx[i]]), float(edges[i][idx[i]+1])]
                       for i in range(n_dims)],
            'density': float(count / vol) if vol > 0 else 0.0,
        })

    sparsity_rank.sort(key=lambda x: x['n_points'])

    return sparsity_rank[:n_regions]


def _compute_spatial_coverage(
    results: List[Dict],
    dims: Tuple[str, ...] = ('v12', 'v21', 'obj_value'),
    n_bins_per_dim: int = 8,
) -> Tuple[float, Dict]:
    """Compute actual spatial coverage fraction across property space.

    Divides each property dimension into n_bins_per_dim intervals and
    counts how many bins contain at least one data point.

    Returns:
        Tuple of (coverage_fraction, coverage_detail_dict).
    """
    valid = [r for r in results if r.get('success') and r.get('obj_value') is not None]
    n_valid = len(valid)

    coverage_detail: Dict = {
        'n_bins_per_dim': n_bins_per_dim,
        'n_dims': len(dims),
        'n_bins_total': 1,
        'n_bins_occupied': 0,
        'dim_coverage': {},
    }

    if n_valid < 2:
        return 0.0, coverage_detail

    # Extract data
    arrays = []
    for d in dims:
        vals = _to_array(valid, d)
        if len(vals) == 0:
            return 0.0, coverage_detail
        arrays.append(vals)

    points = np.column_stack(arrays)
    n_dims = points.shape[1]

    # Define bin edges per dimension (from data range)
    edges = []
    for i in range(n_dims):
        lo, hi = float(points[:, i].min()), float(points[:, i].max())
        if hi - lo < 1e-12:
            hi = lo + 1.0
        edges.append(np.linspace(lo, hi, n_bins_per_dim + 1))

    total_bins = n_bins_per_dim ** n_dims
    coverage_detail['n_bins_total'] = total_bins

    # Bin occupancy
    occupied = np.zeros([n_bins_per_dim] * n_dims, dtype=bool)
    for p in points:
        idxs = []
        for i in range(n_dims):
            idx = np.searchsorted(edges[i][1:], p[i], side='left')
            idx = min(idx, n_bins_per_dim - 1)
            idxs.append(idx)
        occupied[tuple(idxs)] = True

    n_occupied = int(occupied.sum())
    coverage_detail['n_bins_occupied'] = n_occupied

    if total_bins > 0:
        coverage_frac = n_occupied / total_bins
    else:
        coverage_frac = 0.0

    # Per-dim marginal coverage
    for i, d in enumerate(dims):
        # Project: collapse all other dimensions
        axis = i
        projected = np.any(occupied, axis=tuple(
            [j for j in range(n_dims) if j != axis]
        ))
        cov_dim = int(projected.sum()) / n_bins_per_dim
        coverage_detail['dim_coverage'][d] = round(cov_dim, 4)

    return round(coverage_frac, 4), coverage_detail


def coverage_report(
    results: List[Dict],
    dims: Tuple[str, ...] = ('v12', 'v21', 'obj_value'),
    density_result: Optional[Dict] = None,
) -> Dict:
    """Generate a structured coverage quality report.

    Computes TWO coverage metrics:
      - success_rate: fraction of sample runs that succeeded
      - spatial_coverage_pct: fraction of property-space bins occupied
        (true measure of how well the space is explored)

    Args:
        results: List of result dicts.
        dims: Property dimensions to analyze.
        density_result: Pre-computed density (optional).

    Returns:
        Dict with coverage metrics.
    """
    valid = [r for r in results if r.get('success') and r.get('obj_value') is not None]
    n = len(valid)

    # Compute spatial coverage
    spat_frac, spat_detail = _compute_spatial_coverage(valid, dims)

    report: Dict = {
        'n_valid': n,
        'n_total': len(results),
        'success_rate_pct': round(100.0 * n / len(results), 1) if results else 0,
        'spatial_coverage_pct': round(spat_frac * 100, 1),
        'spatial_coverage_detail': spat_detail,
        'dimensions': list(dims),
        'ranges': {},
        'sparsity': {},
        'interpretation': None,
    }

    # Human-readable interpretation
    if spat_frac < 0.3:
        report['interpretation'] = 'LOW spatial coverage — property space is poorly explored'
    elif spat_frac < 0.7:
        report['interpretation'] = 'MODERATE spatial coverage — significant gaps remain'
    else:
        report['interpretation'] = 'HIGH spatial coverage — well-explored property space'

    if n == 0:
        return report

    # Per-dimension coverage
    for d in dims:
        vals = _to_array(valid, d)
        if len(vals) > 0:
            report['ranges'][d] = {
                'min': float(vals.min()),
                'max': float(vals.max()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
            }

    # Sparse regions
    sparse = find_sparse_regions(valid, dims[:2])
    report['sparsity'] = {
        'n_sparse_regions': len(sparse),
        'regions': sparse,
    }

    # Pairwise correlation
    if n >= 5 and len(dims) >= 2:
        corr_matrix = {}
        for i, d1 in enumerate(dims):
            for d2 in dims[i+1:]:
                v1 = _to_array(valid, d1)
                v2 = _to_array(valid, d2)
                if len(v1) > 1 and len(v2) > 1:
                    min_len = min(len(v1), len(v2))
                    c = np.corrcoef(v1[:min_len], v2[:min_len])[0, 1]
                    corr_matrix[f'{d1}_vs_{d2}'] = float(c)
        report['correlations'] = corr_matrix

    # roadmap 6.2/6.3 - chỉ tính trên kết quả CÓ dữ liệu manufacturability
    # (results cũ/mock trước khi thêm instrumentation này có passes_all=None
    # hoặc thiếu hẳn field - bỏ qua thay vì coi là 0/False, tránh làm sai
    # lệch số liệu).
    with_manuf = [r for r in valid if r.get('passes_all') is not None]
    if with_manuf:
        report['manufacturability'] = {
            'n_with_data': len(with_manuf),
            'frac_manufacturable': round(
                float(np.mean([bool(r['passes_all']) for r in with_manuf])), 4
            ),
            'frac_connected': round(
                float(np.mean([bool(r['is_connected']) for r in with_manuf])), 4
            ),
            'frac_periodic': round(
                float(np.mean([bool(r['periodic_ok']) for r in with_manuf])), 4
            ),
        }

    return report


def seed_manufacturability_report(results: List[Dict]) -> Dict[str, Dict]:
    """Per-seed auxetic rate, manufacturable rate, và joint rate (auxetic
    VÀ manufacturable đồng thời) - dùng bởi adaptive.py để quyết định
    trọng số seed. Bỏ qua kết quả thiếu 'passes_all' (dữ liệu cũ/mock chưa
    có instrumentation này) khỏi phần manufacturability, nhưng vẫn tính
    auxetic_rate nếu có v12 (không phụ thuộc lẫn nhau).

    Returns:
        Dict seed -> {n, auxetic_rate, manufacturable_rate, joint_rate,
        n_with_manuf_data}.
    """
    by_seed: Dict[str, List[Dict]] = {}
    for r in results:
        if not r.get('success') or r.get('v12') is None:
            continue
        seed = r.get('seed')
        if not seed:
            continue
        by_seed.setdefault(seed, []).append(r)

    report: Dict[str, Dict] = {}
    for seed, rows in by_seed.items():
        n = len(rows)
        auxetic_rate = float(np.mean([r['v12'] < 0 for r in rows]))

        with_manuf = [r for r in rows if r.get('passes_all') is not None]
        if with_manuf:
            manufacturable_rate = float(np.mean([bool(r['passes_all']) for r in with_manuf]))
            joint_rate = float(np.mean([
                bool(r['passes_all']) and r['v12'] < 0 for r in with_manuf
            ]))
        else:
            manufacturable_rate = None
            joint_rate = None

        report[seed] = {
            'n': n,
            'auxetic_rate': round(auxetic_rate, 4),
            'manufacturable_rate': round(manufacturable_rate, 4) if manufacturable_rate is not None else None,
            'joint_rate': round(joint_rate, 4) if joint_rate is not None else None,
            'n_with_manuf_data': len(with_manuf),
        }

    return report


def recommend_new_samples(
    all_results: List[Dict],
    sparse_regions: List[Dict],
    param_space: Dict[str, Tuple[float, float]],
    n_recommend: int = 20,
    dims: Tuple[str, ...] = ('v12', 'v21'),
) -> List[Dict]:
    """Convert sparse property-space regions back to parameter-space hints.

    This is a heuristic: it finds successful samples NEAR sparse regions
    and suggests variations around their parameter values.

    Args:
        all_results: All batch results so far.
        sparse_regions: Output from find_sparse_regions().
        param_space: Dict of param_name -> (min, max).
        n_recommend: Max number of recommendations.
        dims: The property dimensions used for sparsity.

    Returns:
        List of recommended parameter dicts.
    """
    valid = [r for r in all_results
             if r.get('success') and r.get('obj_value') is not None]
    if not valid or not sparse_regions:
        return []

    recommendations: List[Dict] = []

    for region in sparse_regions[:3]:  # Top 3 sparse regions
        region_bounds = region.get('bounds', [])
        if not region_bounds:
            continue

        # Find points near/inside this region
        near = []
        for r in valid:
            inside = True
            for i, d in enumerate(dims[:len(region_bounds)]):
                val = r.get(d)
                if val is None:
                    inside = False
                    break
                lo, hi = region_bounds[i]
                margin = 0.2 * (hi - lo)
                if not (lo - margin <= val <= hi + margin):
                    inside = False
                    break
            if inside:
                near.append(r)

        if not near:
            # No nearby points; sample middle of region with random params
            from copy import deepcopy
            for _ in range(min(n_recommend // 3, 5)):
                rec = {}
                for pname, (pmin, pmax) in param_space.items():
                    rec[pname] = float(np.random.uniform(pmin, pmax))
                rec['_source'] = 'sparse region (no nearby)'
                # Add property-space centroid as target
                for i, d in enumerate(dims[:len(region_bounds)]):
                    rec[f'_target_{d}'] = float(np.mean(region_bounds[i]))
                recommendations.append(rec)
            continue

        # Perturb parameters of nearby samples
        weights = np.array([r.get('v12', 0) for r in near])
        weights = np.maximum(-weights, 1e-10)
        weights = weights / weights.sum()

        n_from_region = max(2, int(n_recommend / len(sparse_regions[:3])))
        for _ in range(n_from_region):
            idx = np.random.choice(len(near), p=weights)
            base = near[idx]
            rec = {}
            for pname, (pmin, pmax) in param_space.items():
                bval = base['params'].get(pname)
                if bval is None:
                    bval = float(np.random.uniform(pmin, pmax))
                else:
                    bval = float(bval)
                # Jitter
                span = pmax - pmin
                jitter = float(np.random.uniform(-0.1, 0.1) * span)
                rec[pname] = np.clip(bval + jitter, pmin, pmax)
            rec['_source'] = 'near sparse region'
            for i, d in enumerate(dims[:len(region_bounds)]):
                rec[f'_target_{d}'] = float(np.mean(region_bounds[i]))
            recommendations.append(rec)

    return recommendations[:n_recommend]