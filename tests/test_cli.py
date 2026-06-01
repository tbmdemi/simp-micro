"""Tests for the SIMP CLI parameter overrides."""

from simp.main import build_parser, build_params


def test_build_parser_has_expected_options():
    parser = build_parser()
    args = parser.parse_args([
        '--nelx', '60',
        '--nely', '40',
        '--volfrac', '0.3',
        '--seed', 'hexagonal',
        '--objective', 'second',
        '--output_dir', 'outputs/test_cli',
    ])

    assert args.nelx == 60
    assert args.nely == 40
    assert args.volfrac == 0.3
    assert args.seed == 'hexagonal'
    assert args.objective == 'second'
    assert args.output_dir == 'outputs/test_cli'


def test_build_params_applies_cli_overrides():
    parser = build_parser()
    args = parser.parse_args([
        '--nelx', '80',
        '--max_iter', '10',
        '--save_every', '2',
    ])
    params = build_params(args)

    assert params['nelx'] == 80
    assert params['max_iter'] == 10
    assert params['save_every'] == 2
    assert params['nely'] == 100
    assert params['volfrac'] == 0.4
