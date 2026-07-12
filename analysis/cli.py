"""
Command-line interface for the SIMP analysis pipeline.

Provides commands to analyze convergence data, compute image quality
metrics, and generate HTML reports.

Usage:
    python -m analysis.cli report --data-dir outputs/pipeline/phase1/circle/auxetic
    python -m analysis.cli image-metrics --dir outputs/pipeline/phase1/circle/auxetic/sample_0000
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .dataset import build_classification_table
from .image import analyze_image_directory
from .report import generate_html_report


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the analysis CLI.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='SIMP Analysis Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate HTML analysis report',
    )
    report_parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Directory containing SIMP result subdirectories',
    )
    report_parser.add_argument(
        '--output', type=str, default='outputs/report_simp_analysis.html',
        help='Output HTML file path',
    )
    report_parser.add_argument(
        '--objective', type=str, choices=['auxetic'], default='auxetic',
        help='Objective function type (chỉ hỗ trợ auxetic)',
    )
    report_parser.add_argument(
        '--title', type=str, default='SIMP Analysis Report',
        help='Report title',
    )

    # Image metrics command
    img_parser = subparsers.add_parser(
        'image-metrics',
        help='Compute image quality metrics for a directory',
    )
    img_parser.add_argument(
        '--dir', type=str, required=True,
        help='Directory containing iteration images',
    )
    img_parser.add_argument(
        '--pattern', type=str, default='iteration_*.png',
        help='Glob pattern for image files',
    )
    img_parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV file path (optional)',
    )

    # Global options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose (DEBUG) logging')

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == 'report':
        _handle_report(args)
    elif args.command == 'image-metrics':
        _handle_image_metrics(args)
    else:
        parser.print_help()


def _handle_report(args) -> None:
    """Handle the 'report' command."""
    logger = logging.getLogger(__name__)

    data_dir = args.data_dir
    if not Path(data_dir).exists():
        logger.error('Data directory not found: %s', data_dir)
        sys.exit(1)

    logger.info('Building classification table from: %s', data_dir)
    class_df = build_classification_table(data_dir, args.objective)

    if class_df.empty:
        logger.warning('No classification data found.')
    else:
        n_auxetic = len(class_df[class_df['Classification'] == 'Auxetic'])
        logger.info(
            'Found %d shapes (%d auxetic, %d conventional)',
            len(class_df), n_auxetic,
            len(class_df) - n_auxetic,
        )

    # Build image metrics from the first subdirectory with images
    img_df = _find_and_analyze_images(data_dir)

    output_path = generate_html_report(
        classification_df=class_df,
        image_metrics_df=img_df,
        output_path=args.output,
        title=args.title,
    )
    logger.info('Report generated: %s', output_path)


def _handle_image_metrics(args) -> None:
    """Handle the 'image-metrics' command."""
    logger = logging.getLogger(__name__)

    img_dir = args.dir
    if not Path(img_dir).exists():
        logger.error('Image directory not found: %s', img_dir)
        sys.exit(1)

    logger.info('Analyzing images in: %s', img_dir)
    df = analyze_image_directory(img_dir, args.pattern)

    if df.empty:
        logger.warning('No images found.')
        return

    logger.info('Analyzed %d images', len(df))

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info('Metrics saved to: %s', args.output)
    else:
        print(df.to_string(index=False))


def _find_and_analyze_images(data_dir: str) -> pd.DataFrame:
    """Find the first subdirectory with images and analyze them.

    Args:
        data_dir: Root data directory.

    Returns:
        DataFrame with image metrics, or empty DataFrame if no images found.
    """
    data_path = Path(data_dir)
    subdirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and list(d.glob('iteration_*.png'))
    ])

    if not subdirs:
        return pd.DataFrame()

    # Analyze the first subdirectory with images
    return analyze_image_directory(str(subdirs[0]))


if __name__ == '__main__':
    main()
