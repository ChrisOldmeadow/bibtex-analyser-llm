#!/usr/bin/env python3
"""
Run theme search pipeline for strategic research analysis.

Executes hybrid semantic search across all defined themes,
applies LLM relevance filtering, calculates SCImago-style scores,
and generates staff aggregations.

Usage:
    python scripts/run_theme_search.py \\
        --dataset data/institutional_publications.csv \\
        --themes themes.yaml \\
        --baselines data/dataset_baselines.json \\
        --output results/theme_analysis_2025/
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from theme_analysis.pipeline import ThemeSearchPipeline
from theme_analysis.baselines import load_baselines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_publication_data(input_path: Path, deduplicate: bool = True) -> pd.DataFrame:
    """Load institutional publication data from CSV with optional deduplication."""
    logger.info(f"Loading publication data from {input_path}")
    logger.info("  Deduplication: %s", "enabled" if deduplicate else "skipped")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Use BibtexProcessor for consistent loading and deduplication
    from bibtex_analyzer.bibtex_processor import BibtexProcessor

    processor = BibtexProcessor()
    entries = processor.load_entries(str(input_path), deduplicate=deduplicate)

    # Convert to DataFrame
    df = pd.DataFrame(entries)

    logger.info(f"  Loaded {len(df)} publications")
    dedup_stats = processor.deduplication_stats
    if deduplicate:
        logger.info(f"  Original entries: {dedup_stats['total_entries']}")
        logger.info(f"  Duplicates removed: {dedup_stats['duplicates_removed']}")
        if dedup_stats['staff_contributors']:
            multi_author_count = sum(1 for staff_list in dedup_stats['staff_contributors'].values() if len(staff_list) > 1)
            logger.info(f"  Multi-staff publications with >1 staff: {multi_author_count}")

    return df


def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.error("Please set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    logger.info("✅ OpenAI API key found")


def validate_themes_file(themes_path: Path):
    """Validate themes YAML file exists and is readable."""
    if not themes_path.exists():
        raise FileNotFoundError(f"Themes file not found: {themes_path}")

    import yaml
    with open(themes_path, 'r') as f:
        config = yaml.safe_load(f)

    themes = config.get('themes', [])
    if not themes:
        raise ValueError(f"No themes found in {themes_path}")

    logger.info(f"✅ Found {len(themes)} themes in {themes_path}")

    # Validate theme structure
    required_fields = ['id', 'name', 'narrative']
    for i, theme in enumerate(themes):
        for field in required_fields:
            if field not in theme:
                raise ValueError(f"Theme {i+1} missing required field '{field}'")

    return themes


def main():
    parser = argparse.ArgumentParser(
        description="Run theme search pipeline for strategic research analysis"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to institutional publications CSV'
    )
    parser.add_argument(
        '--themes',
        type=Path,
        required=True,
        help='Path to themes YAML file'
    )
    parser.add_argument(
        '--baselines',
        type=Path,
        required=False,
        help='Optional path to baselines JSON (from calculate_baselines.py). If not provided, will use OpenAlex enrichment data where available.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--min-year',
        type=int,
        help='Minimum publication year to include (optional)'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        help='Maximum publication year to include (optional)'
    )
    parser.add_argument(
        '--skip-dedup',
        action='store_true',
        help='Skip deduplication when loading the dataset'
    )
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=None,
        help='Override the per-theme max_candidates limit for LLM reranking (default 100 per theme unless set in YAML)'
    )
    parser.add_argument(
        '--semantic-only',
        action='store_true',
        help='Skip the LLM rerank step and rely on semantic embeddings only (faster, no GPT cost)'
    )
    parser.add_argument(
        '--candidate-prompt',
        action='store_true',
        help='Prompt before limiting to max_candidates when far more embedding matches are found'
    )
    parser.add_argument(
        '--ignore-max-limits',
        action='store_true',
        help='Ignore per-theme max_candidates/max_results values and use all matches'
    )
    parser.add_argument(
        '--semantic-threshold',
        type=float,
        default=None,
        help='Override semantic threshold for all themes (0-1, default: use each theme setting)'
    )

    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info("THEME SEARCH PIPELINE")
    logger.info("="*60 + "\n")

    try:
        # Check prerequisites
        check_api_key()

        # Validate inputs
        themes = validate_themes_file(args.themes)

        # Load baselines (optional)
        if args.baselines:
            logger.info("Loading baselines...")
            baselines = load_baselines(args.baselines)
        else:
            logger.info("No baselines provided - will use OpenAlex enrichment data where available")
            baselines = None

        # Load publication data
        df = load_publication_data(args.dataset, deduplicate=not args.skip_dedup)

        # Filter by year range if specified
        if args.min_year or args.max_year:
            original_len = len(df)
            if args.min_year:
                df = df[df['Reported_Year'] >= args.min_year]
                logger.info(f"  Filtered to publications >= {args.min_year}")
            if args.max_year:
                df = df[df['Reported_Year'] <= args.max_year]
                logger.info(f"  Filtered to publications <= {args.max_year}")
            logger.info(f"  {len(df)} publications after year filtering (was {original_len})")

        # Initialize pipeline
        logger.info("\nInitializing theme search pipeline...")
        pipeline = ThemeSearchPipeline(
            df,
            baselines,
            max_candidates=args.max_candidates,
            semantic_only=args.semantic_only,
            prompt_on_overflow=args.candidate_prompt,
            ignore_max_limits=args.ignore_max_limits,
            global_semantic_threshold=args.semantic_threshold,
        )

        # Run all themes
        logger.info("\nStarting theme search...\n")
        results = pipeline.run_all_themes(args.themes, args.output)

        # Success summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"✅ Processed {results['themes_processed']} themes")
        logger.info(f"✅ Results saved to: {results['output_dir']}")
        if results['comparison_file']:
            logger.info(f"✅ Theme comparison: {results['comparison_file']}")
        logger.info("="*60 + "\n")

        logger.info("Next steps:")
        logger.info(f"  1. Review theme_comparison.csv for overall rankings")
        logger.info(f"  2. Examine individual theme folders for detailed papers")
        logger.info(f"  3. Check staff_summary.csv files to see researcher contributions")
        logger.info("")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
