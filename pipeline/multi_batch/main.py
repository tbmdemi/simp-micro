"""
CLI entry point for multi-batch adaptive sampling pipeline.

Usage:
    python -m pipeline.multi_batch.main --phase1-summary <path> [options]

Workflow:
    1. Load phase 1 summary → extract param ranges, existing results.
    2. Generate batch 1 design with Sobol sequence (100-150 points).
    3. Run batch 1 via SIMP (or parallel runner).
    4. Analyze coverage → decide next action via adaptive.py.
    5. Repeat for subsequent batches until stop condition.
"""

import argparse
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.multi_batch.params import (
    BatchConfig,
    BatchMode,
    SamplingStrategy,
    load_phase1_params,
    prepare_output,
)
from pipeline.multi_batch.sampling import generate_design
from pipeline.multi_batch.runner import run_batch_from_design
from pipeline.multi_batch.coverage import (
    coverage_report,
    find_sparse_regions,
)
from pipeline.multi_batch.adaptive import (
    decide_next_action,
    _load_accumulated_results,
)
from pipeline.multi_batch.visualize import (
    generate_coverage_html,
    generate_batch_progression_html,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Multi-batch adaptive sampling pipeline for SIMP.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--phase1-summary',
        type=str,
        required=True,
        help='Path to phase 1 summary JSON file (or phase1 output dir).',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='outputs/multi_batch',
        help='Root output directory for batch results.',
    )
    parser.add_argument(
        '--n-batch1',
        type=int,
        default=120,
        help='Number of samples in batch 1 (default: 120).',
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=5,
        help='Maximum number of batches to run (default: 5).',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['sobol', 'lhs', 'optimized_lhs'],
        default='sobol',
        help='Sampling strategy for batch 1 (default: sobol).',
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip actual SIMP execution (dry-run mode for testing).',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a decision log JSON to resume from previous run.',
    )
    parser.add_argument(
        '--only-report',
        action='store_true',
        help='Only generate HTML reports from existing results, do not run new batches.',
    )
    parser.add_argument(
        '--seeds',
        type=str,
        nargs='+',
        default=None,
        help='Seeds to use for the batch (default: first 5 seeds).',
    )
    parser.add_argument(
        '--objectives',
        type=str,
        nargs='+',
        choices=['auxetic', 'first', 'second'],
        default=None,
        help='Objectives for the batch (default: auxetic + first).',
    )

    return parser.parse_args()


def _find_summary_jsons(phase1_path: str) -> List[str]:
    """Find all summary JSON files in the given path (file or directory)."""
    p = Path(phase1_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        candidates = []
        for pattern in ['*summary*.json', '*_results.json', '*batch*.json']:
            candidates.extend(sorted(p.glob(pattern)))
        return [str(c) for c in candidates]
    return []


def _build_pipeline_config(
    summaries: List[Dict],
    batch_configs: List[BatchConfig],
    active_params_meta: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Build a minimal PipelineConfig-like dict from summaries and configs.

    This is used by decide_next_action which expects a config object
    with attributes: active, active_seeds, active_objectives, batches.

    Args:
        summaries: List of batch summary dicts.
        batch_configs: List of BatchConfig for reference.
        active_params_meta: Pre-extracted active param metadata
            (avoids re-parsing refined_parameters.json).

    Returns:
        A simple Dict with those keys.
    """
    # Collect all used seeds and objectives
    all_seeds = set()
    all_objectives = set()
    all_active: Dict[str, Dict] = {}

    # Use pre-extracted active_params_meta if provided
    if active_params_meta:
        for pname, pdef in active_params_meta.items():
            r = pdef.get('range', pdef.get('bounds', []))
            if len(r) == 2:
                all_active[pname] = {'range': list(r)}
    else:
        # Fall back to parsing summaries
        for s in summaries:
            # Try refined_parameters.json format first
            for key in ('active_parameters', 'parameters'):
                params = s.get(key, {})
                if not isinstance(params, dict):
                    continue
                for pname, pdef in params.items():
                    if isinstance(pdef, dict) and 'range' in pdef:
                        all_active[pname] = {
                            'range': list(pdef['range']),
                        }
                if all_active:
                    break

        combos = s.get('combinations', []) or s.get('best_per_combo', [])
        if isinstance(combos, dict):
            for key in combos:
                parts = key.split('_')
                if len(parts) >= 2:
                    all_seeds.add(parts[0])
                    all_objectives.add(parts[1])

    # If still empty, fatal error
    if not all_active:
        print(
            "FATAL: _build_pipeline_config could not extract active params.\n"
            "  Check that summaries contain either 'active_parameters' or 'parameters' with 'range' key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not all_seeds:
        all_seeds = {'circle', 'square', 'hourglass', 'hexagonal', 'cross_rectangular'}

    if not all_objectives:
        all_objectives = {'auxetic'}

    return {
        'active': all_active,
        'active_seeds': sorted(all_seeds),
        'active_objectives': sorted(all_objectives),
        'batches': batch_configs,
    }


def main() -> None:
    """Main entry point for the multi-batch adaptive pipeline."""
    args = parse_args()

    # ── Step 1: Load phase 1 summary ──
    summary_files = _find_summary_jsons(args.phase1_summary)
    if not summary_files:
        print(f"ERROR: No summary JSON files found at '{args.phase1_summary}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(summary_files)} summary file(s):")
    for sf in summary_files[:5]:
        print(f"  • {sf}")
    if len(summary_files) > 5:
        print(f"  … and {len(summary_files) - 5} more")

    # Load all summaries
    summaries = []
    for sf in summary_files:
        with open(sf) as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    summaries.extend(data)
                else:
                    summaries.append(data)
            except json.JSONDecodeError as e:
                print(f"WARNING: Could not parse {sf}: {e}")

    if not summaries:
        print("ERROR: No valid summary data loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(summaries)} summary record(s).")

    # ── Step 2: Load/initialize batch configs ──
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Detect refined_parameters.json structure (active_parameters/fixed_parameters)
    # vs. generic phase1 summary (parameters key).
    fixed_params: Dict[str, float] = {}
    active_params_meta: Dict[str, Dict] = {}

    for s in summaries:
        if 'active_parameters' in s and 'fixed_parameters' in s:
            # refined_parameters.json format
            active_params_meta.update(s.get('active_parameters', {}))
            fixed_params.update(s.get('fixed_parameters', {}))
        elif 'parameters' in s:
            # generic phase1 summary format
            for pname, pdef in s['parameters'].items():
                if isinstance(pdef, dict) and 'range' in pdef:
                    active_params_meta[pname] = pdef

    # Build param_ranges from active_params_meta
    param_ranges: Dict[str, Tuple[float, float]] = {}
    if active_params_meta:
        for pname, pdef in active_params_meta.items():
            r = pdef.get('range', pdef.get('bounds', []))
            if len(r) == 2:
                param_ranges[pname] = (float(r[0]), float(r[1]))
    else:
        print(
            "FATAL: No active parameters found in phase 1 summary.\n"
            f"  Looked for keys 'active_parameters' or 'parameters' in {len(summaries)} summary file(s).\n"
            "  Ensure refined_parameters.json has the correct structure:\n"
            "    { 'active_parameters': { '<name>': { 'range': [lo, hi] } }, 'fixed_parameters': { ... } }",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Using {len(param_ranges)} active parameter(s).")
    if fixed_params:
        print(f"Fixed parameters: {fixed_params}")

    # Seeds & objectives
    seeds = ['circle', 'square', 'hourglass', 'hexagonal', 'cross_rectangular',
             'nine_circle', 'four_circle', 'grid_circular_voids',
             'small_square_cross', 'circle_half_quarter']
    objectives = ['auxetic', 'first', 'second']

    # ── Step 3: Decision loop ──
    decision_log_path = output_root / 'decision_log.json'
    batch_summaries: List[Dict] = []

    # Load existing decision log if resuming
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                saved = json.load(f)
            batch_summaries = saved.get('batch_summaries', [])
            print(f"Resuming with {len(batch_summaries)} existing batch summary(ies).")

    # Strategy mapping
    strategy_map = {
        'sobol': SamplingStrategy.SOBOL,
        'lhs': SamplingStrategy.LHS,
        'optimized_lhs': SamplingStrategy.OPTIMIZED_LHS,
    }

    current_batch_id = len(batch_summaries) + 1
    n_to_run = args.max_batches - len(batch_summaries)

    # Initial batch config (will be overwritten for batch 1)
    batch_config = None

    for batch_idx in range(n_to_run):
        batch_id = current_batch_id + batch_idx
        print(f"\n{'='*60}")
        print(f"  BATCH {batch_id} / up to {args.max_batches}")
        print(f"{'='*60}")

        # Build config for this batch
        if batch_id == 1:
            strat = strategy_map.get(args.strategy, SamplingStrategy.SOBOL)
            batch_seeds = args.seeds if args.seeds is not None else seeds[:5]
            batch_objectives = args.objectives if args.objectives is not None else objectives[:2]
            batch_config = BatchConfig(
                batch_id=batch_id,
                n_samples=args.n_batch1,
                strategy=strat,
                mode=BatchMode.EXPLORE,
                objectives=batch_objectives,
                seeds=batch_seeds,
                param_ranges=param_ranges,
            )
        else:
            # Use decide_next_action to determine next config
            # Build a simple PipelineConfig-like dict
            pipeline_cfg = _build_pipeline_config(
                batch_summaries,
                [batch_config],  # type: ignore  # noqa: F821 — only used for ref
                active_params_meta=active_params_meta,
            )

            decision = decide_next_action(
                summaries=batch_summaries,
                config=pipeline_cfg,
                max_batches=args.max_batches,
                param_ranges=param_ranges,
            )

            action = decision.get('action', 'stop')
            reason = decision.get('reason', '')
            print(f"  Decision: {action}")
            print(f"  Reason: {reason}")

            if action == 'stop':
                print("\n✅ Stopping: no further batches needed.")
                break

            decision_config = decision.get('next_config')
            if decision_config:
                batch_config = decision_config
                # Override param_ranges if decision narrowed them
                narrowed = getattr(decision_config, 'narrowed_params', None)
                if narrowed:
                    print(f"  Using narrowed parameter ranges ({len(narrowed)} params).")
                    for pname, pdef in narrowed.items():
                        if pdef['range'] != param_ranges.get(pname):
                            print(f"    {pname}: [{pdef['range'][0]:.4f}, {pdef['range'][1]:.4f}]")
            else:
                # Fallback: repeat previous config with +20% samples
                batch_config = BatchConfig(
                    batch_id=batch_id,
                    n_samples=int(batch_config.n_samples * 1.2),  # type: ignore  # noqa: F821
                    strategy=batch_config.strategy,  # type: ignore  # noqa: F821
                    mode=BatchMode.EXPLORE,
                    objectives=objectives[:2],
                    seeds=seeds[:5],
                    param_ranges=param_ranges,
                )

            # Save decision (JSON-safe)
            decision_record = {
                'batch_id': batch_id,
                'action': action,
                'reason': reason,
                'decision_detail': decision,
            }
            with open(output_root / f'decision_batch{batch_id}.json', 'w') as f:
                json.dump(decision_record, f, indent=2, cls=_JSONEncoder)

        # ── Step 3a: Generate design ──
        print(f"\n  Generating {batch_config.n_samples} samples via {batch_config.strategy.value}…")

        design = generate_design(
            n_samples=batch_config.n_samples,
            param_ranges=batch_config.param_ranges,
            strategy=batch_config.strategy,
            batch_id=batch_config.batch_id,
            seed_map=batch_config.seeds,
            objective_map=batch_config.objectives,
        )

        print(f"  Design has {len(design)} rows.")

        # ── Step 3b: Save design ──
        batch_output_dir = output_root / f'batch_{batch_id}'
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        design_path = batch_output_dir / f'batch_{batch_id}_design.csv'
        design.to_csv(design_path, index=False)
        print(f"  Design saved to {design_path}")

        # Save params as JSON
        params_path = batch_output_dir / f'batch_{batch_id}_params.json'
        batch_config_dict = {
            'batch_id': batch_config.batch_id,
            'n_samples': batch_config.n_samples,
            'strategy': batch_config.strategy.value,
            'mode': batch_config.mode.value,
            'objectives': batch_config.objectives,
            'seeds': batch_config.seeds,
            'param_ranges': {k: list(v) for k, v in batch_config.param_ranges.items()},
        }
        with open(params_path, 'w') as f:
            json.dump(batch_config_dict, f, indent=2)

        # ── Step 3c: Run batch ──
        if args.skip_run or args.only_report:
            print("  SKIP_RUN: skipping SIMP execution.")
            # Generate mock results for testing
            batch_summary = _mock_batch_summary(batch_config, design, batch_output_dir)
        else:
            print("\n  Running SIMP batch…")
            try:
                batch_summary = run_batch_from_design(
                    design=design,
                    output_dir=str(batch_output_dir),
                    batch_id=batch_config.batch_id,
                    n_workers=4,
                )
            except Exception as e:
                print(f"  ERROR during batch execution: {e}", file=sys.stderr)
                print("  Saving partial design and exiting.")
                batch_summary = {
                    'batch_id': batch_config.batch_id,
                    'n_samples': len(design),
                    'status': 'failed',
                    'error': str(e),
                    'output_dir': str(batch_output_dir),
                }

        # Save batch summary
        summary_path = batch_output_dir / f'batch_{batch_id}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        batch_summaries.append(batch_summary)

        # ── Step 3d: Coverage analysis ──
        print("\n  Analyzing coverage…")
        all_results = _load_accumulated_results(batch_summaries)

        sparse_regions = find_sparse_regions(all_results)
        cov = coverage_report(all_results)

        print(f"  Coverage: {cov.get('coverage_pct', 0):.1f}% valid")
        print(f"  Sparse regions: {cov.get('sparsity', {}).get('n_sparse_regions', 0)}")

        # ── Step 3e: Generate visual reports ──
        print("  Generating HTML reports…")
        vis_dir = output_root / 'reports'
        vis_dir.mkdir(parents=True, exist_ok=True)

        coverage_page = generate_coverage_html(
            output_dir=str(vis_dir),
            all_results=all_results,
            sparse_regions=sparse_regions,
            title=f'Coverage Analysis — After Batch {batch_id}',
        )
        print(f"  Coverage plot: {coverage_page}")

        # Batch progression (if >1 batch)
        if len(batch_summaries) > 1:
            batch_dirs = [
                str(output_root / f'batch_{s["batch_id"]}')
                for s in batch_summaries
                if os.path.isdir(str(output_root / f'batch_{s["batch_id"]}'))
            ]
            if batch_dirs:
                prog_page = generate_batch_progression_html(
                    output_dir=str(vis_dir),
                    batch_dirs=batch_dirs,
                )
                print(f"  Progression plot: {prog_page}")

        # Save consolidated decision log
        log_entry = {
            'batch_summaries': batch_summaries,
            'coverage': cov,
            'sparse_regions': sparse_regions,
            'last_batch_id': batch_id,
        }
        with open(decision_log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)

        if args.only_report:
            print("\n✅ Only-report mode: generated reports without running new batches.")
            break

    else:
        # Loop completed without break
        print(f"\n✅ Completed all {args.max_batches} batches.")

    # ── Final report ──
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Batches completed: {len(batch_summaries)}")
    print(f"  Decision log: {decision_log_path}")
    print(f"  Reports: {vis_dir}")
    print(f"{'='*60}")


class _JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enums, BatchConfig, and dataclasses."""
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, '__dataclass_fields__'):
            return {k: getattr(o, k) for k in o.__dataclass_fields__}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _mock_batch_summary(
    config: BatchConfig,
    design: 'pd.DataFrame',
    output_dir: Path,
) -> Dict:
    """Generate mock summary for testing without running SIMP."""
    import pandas as pd

    n = len(design)
    np.random.seed(42 + config.batch_id)

    # Mock results
    results = []
    for idx, row in design.iterrows():
        v12 = np.random.uniform(-0.8, 0.5)
        v21 = np.random.uniform(-0.8, 0.5)
        obj = np.random.uniform(0.01, 0.5)
        results.append({
            'sample_id': int(idx),
            'success': np.random.random() > 0.1,
            'seed': str(row.get('seed', 'circle')),
            'objective': str(row.get('objective', 'auxetic')),
            'params': {k: float(row[k]) for k in config.param_ranges if k in row},
            'v12': v12,
            'v21': v21,
            'obj_value': obj,
        })

    # Save mock results
    results_path = os.path.join(str(output_dir), f'batch_{config.batch_id}_results.json')
    with open(results_path, 'w') as f:
        json.dump({'results': results}, f, indent=2)

    summary = {
        'batch_id': config.batch_id,
        'n_samples': n,
        'n_success': sum(1 for r in results if r['success']),
        'status': 'completed_mock',
        'output_dir': str(output_dir),
        'best_per_combo': {
            'circle_auxetic': {'obj_value': min(
                r['obj_value'] for r in results if r['success']
            )},
        },
        'parameters': {k: {'range': list(v)} for k, v in config.param_ranges.items()},
    }
    return summary


if __name__ == '__main__':
    main()