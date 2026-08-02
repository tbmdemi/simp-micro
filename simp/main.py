"""Console entry point for SIMP optimization."""

import argparse
from typing import Any

from .run import params as default_params, run_simp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run SIMP topology optimization with optional parameter overrides.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--version', action='store_true', help='Show version and exit')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--list-seeds', action='store_true', help='List available seed patterns and exit')

    parser.add_argument('--nelx', type=int, help='Number of elements in the x direction')
    parser.add_argument('--nely', type=int, help='Number of elements in the y direction')
    parser.add_argument('--volfrac', type=float, help='Target volume fraction')
    parser.add_argument('--penal', type=float, help='Penalization factor')
    parser.add_argument('--rmin', type=float, help='Filter radius')
    parser.add_argument('--ft', type=int, choices=[1, 2], help='Filter type')
    parser.add_argument('--E0', type=float, help='Young-modulus of solid material')
    parser.add_argument('--Emin', type=float, help='Young-modulus of void material')
    parser.add_argument('--nu', type=float, help='Poisson ratio for the base material')
    parser.add_argument('--move', type=float, help='Move limit for OC update')
    parser.add_argument('--move_min', type=float,
                         help='Experimental: linearly decay --move down to this floor '
                              'over max_iter iterations (default: unset, move stays fixed)')
    parser.add_argument('--penal_init', type=float,
                         help='Experimental: linearly ramp --penal up from this starting value '
                              'over max_iter iterations (continuation, default: unset, penal stays fixed)')
    parser.add_argument('--use_sqrt', action='store_true',
                         help='Experimental: use classic OC damping exponent eta=0.5 '
                              '(x*sqrt(ratio)) instead of x*ratio (default: off)')
    parser.add_argument('--projection', type=str, choices=['heaviside'],
                         help='Experimental: robust/minimum-length-scale Heaviside projection '
                              '(Wang-Lazarov-Sigmund 2011), only supported with --ft 2 '
                              '(default: unset, off - identity, same as before)')
    parser.add_argument('--beta_proj', type=float,
                         help='Heaviside projection sharpness (only used with --projection heaviside, default 8.0)')
    parser.add_argument('--beta_proj_init', type=float,
                         help='Experimental: linearly ramp --beta_proj up from this starting value over '
                              'max_iter iterations (continuation, only used with --projection heaviside, '
                              'default: unset, beta_proj stays fixed)')
    parser.add_argument('--eta', type=float,
                         help='Heaviside projection threshold (only used with --projection heaviside, default 0.5)')
    parser.add_argument('--enforce_connectivity', action='store_true',
                         help='Experimental: every --connectivity_every iterations, nudge disconnected '
                              'material islands toward void (default: off)')
    parser.add_argument('--connectivity_every', type=int,
                         help='Iteration interval for --enforce_connectivity (default 10)')
    parser.add_argument('--connectivity_floor', type=float,
                         help='Design-variable floor value for nudged island pixels (default 0.01)')
    parser.add_argument('--connectivity_min_relative_size', type=float,
                         help='Fraction of the largest component size below which a disconnected '
                              'island is treated as noise and nudged (default 0.15); larger islands '
                              'are left alone to allow natural reconnection')
    parser.add_argument('--objective_variant', type=str, choices=['q12', 'normalized'],
                         help="Experimental: 'normalized' minimizes Q12/sqrt(Q11*Q22) instead of "
                              "raw Q12 (default: unset, 'q12' - same as before)")
    parser.add_argument('--stiffness_mode', type=str, choices=['gate', 'dual'],
                         help="Experimental: 'dual' uses a two-multiplier OC update (separate "
                              "Lagrange multiplier for the Q11/Q22 stiffness constraint, nested "
                              "bisection) instead of the binary stiff_ok gate on the volume "
                              "bisection (default: unset, 'gate' - same as before). See "
                              "core/oc.py::oc_update_dual() docstring for why 'gate' can oscillate.")
    parser.add_argument('--max_iter', type=int, help='Maximum number of optimization iterations')
    parser.add_argument('--tol_change', type=float, help='Tolerance for design change convergence')
    parser.add_argument('--tol_obj', type=float, help='Tolerance for objective stability convergence')
    parser.add_argument('--window_size', type=int, help='Window size for objective stability evaluation')
    parser.add_argument('--seed', type=str, help='Seed pattern name for initial design')
    parser.add_argument('--objective', type=str, choices=['auxetic'], help='Objective type (chỉ hỗ trợ auxetic)')
    parser.add_argument('--void_size_frac', type=float, help='Void size fraction for seed generation')
    parser.add_argument('--rotation_deg', type=float, help='Rotation angle for seed generation')
    parser.add_argument('--beta', type=float, help='Beta value for objective scaling')
    parser.add_argument('--save_every', type=int, help='How often to save density image iterations')
    parser.add_argument('--scale_factor', type=int, help='Scale factor for density images')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--quiet', action='store_true', help='Suppress final summary output')

    return parser


_NON_PARAM_ARGS = {'quiet', 'version', 'verbose', 'list_seeds'}


def _iter_cli_overrides(args: argparse.Namespace):
    for key, value in vars(args).items():
        if value is not None and key not in _NON_PARAM_ARGS:
            yield key, value


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    params = default_params.copy()
    for key, value in _iter_cli_overrides(args):
        params[key] = value
    return params


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(__version__)
        return

    if args.list_seeds:
        from .runner import SEED_MAP
        print('\n'.join(sorted(SEED_MAP)))
        return

    params = build_params(args)

    if args.verbose:
        print(f'Running SIMP with seed={params.get("seed")}, objective={params.get("objective")}')
        print(f'  mesh: {params.get("nelx")}x{params.get("nely")}, volfrac={params.get("volfrac")}')

    result = run_simp(params)

    if not args.quiet:
        print(f"\nKết quả cuối: ν₁₂={result['v12']:.4f}, ν₂₁={result['v21']:.4f}")
        print(f"Đầu ra tại: {result['output_dir']}")


if __name__ == '__main__':
    main()
