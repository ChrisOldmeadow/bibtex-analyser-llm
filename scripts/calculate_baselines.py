#!/usr/bin/env python3
"""
Calculate baseline metrics from institutional publication dataset.

Computes field-normalized citation baselines, excellence thresholds,
and altmetric distributions for use in theme scoring.

Usage:
    python scripts/calculate_baselines.py \\
        --input data/institutional_publications.csv \\
        --output data/dataset_baselines.json
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from theme_analysis.baselines import calculate_dataset_baselines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_publication_data(input_path: Path) -> pd.DataFrame:
    """Load institutional publication data from CSV with automatic deduplication."""
    logger.info(f"Loading publication data from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Use BibtexProcessor for consistent loading and deduplication
    from bibtex_analyzer.bibtex_processor import BibtexProcessor

    processor = BibtexProcessor()
    entries = processor.load_entries(str(input_path), deduplicate=True)

    # Convert to DataFrame
    df = pd.DataFrame(entries)

    logger.info(f"  Loaded {len(df)} unique publications (after deduplication)")
    logger.info(f"  Original entries: {processor.deduplication_stats['total_entries']}")
    logger.info(f"  Duplicates removed: {processor.deduplication_stats['duplicates_removed']}")

    if processor.deduplication_stats['staff_contributors']:
        multi_author_count = sum(1 for staff_list in processor.deduplication_stats['staff_contributors'].values() if len(staff_list) > 1)
        logger.info(f"  Multi-staff publications: {multi_author_count}")

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Check for required columns in dataset."""
    required_cols = {
        'Citations_Scopus', 'Citations_WoS',
        'FoR_Codes_2020', 'Reported_Year',
        'Altmetrics_Score'
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        logger.warning(f"Missing columns (will use defaults): {missing_cols}")

    # Check coverage
    logger.info("\nData coverage:")
    for col in required_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            logger.info(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate baseline metrics for theme scoring"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to institutional publications CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to save baselines JSON'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("BASELINE CALCULATION")
    logger.info("="*60)

    try:
        # Load data
        df = load_publication_data(args.input)

        # Validate
        validate_required_columns(df)

        # Calculate baselines
        logger.info("\n" + "="*60)
        logger.info("CALCULATING BASELINES")
        logger.info("="*60 + "\n")

        baselines = calculate_dataset_baselines(df, args.output)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("BASELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total publications: {baselines['metadata']['total_publications']}")
        logger.info(f"Unique FoR codes: {baselines['metadata']['unique_for_codes']}")
        logger.info(f"Year range: {baselines['metadata']['year_range']}")
        logger.info(f"\nCitation baselines:")
        logger.info(f"  FoR/Year/Type groups: {len([k for k in baselines['citations'].keys() if k != 'overall_median'])}")
        logger.info(f"  Overall median: {baselines['citations']['overall_median']:.1f}")
        logger.info(f"\nExcellence thresholds:")
        logger.info(f"  FoR/Year groups: {len([k for k in baselines['excellence'].keys() if k != 'overall_p90'])}")
        logger.info(f"  Overall 90th percentile: {baselines['excellence']['overall_p90']:.1f}")
        logger.info(f"\nAltmetric distribution:")
        logger.info(f"  95th percentile: {baselines['altmetrics']['p95']:.1f}")
        logger.info(f"  Coverage: {baselines['altmetrics']['coverage_rate']:.1f}%")
        logger.info(f"\nBaselines saved to: {args.output}")
        logger.info("="*60 + "\n")

        logger.info("✅ Baseline calculation complete!")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
