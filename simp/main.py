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
    parser.add_argument('--max_iter', type=int, help='Maximum number of optimization iterations')
    parser.add_argument('--tol_change', type=float, help='Tolerance for design change convergence')
    parser.add_argument('--tol_obj', type=float, help='Tolerance for objective stability convergence')
    parser.add_argument('--window_size', type=int, help='Window size for objective stability evaluation')
    parser.add_argument('--seed', type=str, help='Seed pattern name for initial design')
    parser.add_argument('--objective', type=str, choices=['first', 'second', 'auxetic'], help='Objective type')
    parser.add_argument('--void_size_frac', type=float, help='Void size fraction for seed generation')
    parser.add_argument('--rotation_deg', type=float, help='Rotation angle for seed generation')
    parser.add_argument('--beta', type=float, help='Beta value for objective scaling')
    parser.add_argument('--beta_second', type=float, help='Beta value for second objective scaling')
    parser.add_argument('--save_every', type=int, help='How often to save density image iterations')
    parser.add_argument('--scale_factor', type=int, help='Scale factor for density images')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--quiet', action='store_true', help='Suppress final summary output')

    return parser


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    params = default_params.copy()
    for key, value in vars(args).items():
        if value is not None and key != 'quiet':
            params[key] = value
    return params


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
