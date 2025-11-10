#!/usr/bin/env python3
"""
Analyze hierarchical theme results and rank topics within each parent theme.

Takes the flat theme_comparison.csv output and restructures it to show:
1. Topics ranked within each parent theme
2. Overall theme-level summaries
3. Cross-theme comparisons

Usage:
    python scripts/analyze_hierarchical_themes.py \
        --results results/hmri_themes_2025/theme_comparison.csv \
        --output results/hmri_themes_2025/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_theme_comparison(comparison_file: Path) -> pd.DataFrame:
    """Load theme comparison CSV with all topic results."""
    logger.info(f"Loading theme comparison from {comparison_file}")
    df = pd.read_csv(comparison_file)

    # Extract parent theme and topic number from theme_id
    # Expected format: env_health_1_housing_food_water
    # Or from dedicated columns if present

    if 'parent_theme' not in df.columns:
        logger.info("Extracting parent theme from theme_id...")
        # Try to extract from theme_id (assumes format: {parent}_{topic_num}_{description})
        # This is a fallback - better to have explicit parent_theme in themes.yaml
        df['parent_theme'] = df['theme_id'].str.extract(r'^(.+?)_\d+_')[0]
        df['topic_number'] = df['theme_id'].str.extract(r'_(\d+)_')[0].astype(int)

    logger.info(f"  Loaded {len(df)} topics across {df['parent_theme'].nunique()} parent themes")

    return df


def generate_theme_rankings(df: pd.DataFrame, output_dir: Path):
    """Generate rankings of topics within each parent theme."""

    # Group by parent theme
    parent_themes = df['parent_theme'].unique()

    all_rankings = []

    for parent in sorted(parent_themes):
        logger.info(f"\n{'='*60}")
        logger.info(f"PARENT THEME: {parent}")
        logger.info('='*60)

        # Filter to this parent theme
        theme_df = df[df['parent_theme'] == parent].copy()

        # Sort by theme_score descending
        theme_df = theme_df.sort_values('theme_score', ascending=False)

        # Add rank within parent theme
        theme_df['rank_within_theme'] = range(1, len(theme_df) + 1)

        # Display summary
        logger.info(f"\nTopic Rankings (by theme_score):\n")
        for _, row in theme_df.iterrows():
            logger.info(
                f"  {row['rank_within_theme']}. {row['theme_name'][:50]:<50} "
                f"Score: {row['theme_score']:.1f} | "
                f"Pubs: {row['publications']:>3} ({row['publications_scored']:>3} scored) | "
                f"Completeness: {row['data_completeness_rate']:.1f}%"
            )

        # Calculate theme-level summary
        theme_summary = {
            'parent_theme': parent,
            'total_topics': len(theme_df),
            'total_publications': theme_df['publications'].sum(),
            'total_publications_scored': theme_df['publications_scored'].sum(),
            'total_publications_excluded': theme_df['publications_excluded'].sum(),
            'avg_completeness_rate': theme_df['data_completeness_rate'].mean(),
            'avg_theme_score': theme_df['theme_score'].mean(),
            'avg_research_score': theme_df['research_score'].mean(),
            'avg_societal_score': theme_df['societal_score'].mean(),
            'unique_staff_total': theme_df['unique_staff'].sum(),  # May have duplicates across topics
        }

        logger.info(f"\nTheme Summary:")
        logger.info(f"  Total publications: {theme_summary['total_publications']}")
        logger.info(f"  Scored publications: {theme_summary['total_publications_scored']}")
        logger.info(f"  Excluded publications: {theme_summary['total_publications_excluded']}")
        logger.info(f"  Average completeness: {theme_summary['avg_completeness_rate']:.1f}%")
        logger.info(f"  Average theme score: {theme_summary['avg_theme_score']:.1f}")
        logger.info(f"  Average research score: {theme_summary['avg_research_score']:.1f}")
        logger.info(f"  Average societal score: {theme_summary['avg_societal_score']:.1f}")

        all_rankings.append(theme_df)

    # Combine all rankings
    combined_rankings = pd.concat(all_rankings, ignore_index=True)

    # Save rankings by parent theme
    output_file = output_dir / 'topic_rankings_by_theme.csv'
    combined_rankings.to_csv(output_file, index=False)
    logger.info(f"\n✅ Saved topic rankings to: {output_file}")

    return combined_rankings


def generate_theme_summary(df: pd.DataFrame, output_dir: Path):
    """Generate parent theme-level summary."""

    summary_data = []

    for parent in sorted(df['parent_theme'].unique()):
        theme_df = df[df['parent_theme'] == parent]

        summary = {
            'parent_theme': parent,
            'topics_count': len(theme_df),
            'publications_total': theme_df['publications'].sum(),
            'publications_scored': theme_df['publications_scored'].sum(),
            'publications_excluded': theme_df['publications_excluded'].sum(),
            'data_completeness_rate': theme_df['data_completeness_rate'].mean(),
            'theme_score_avg': theme_df['theme_score'].mean(),
            'theme_score_max': theme_df['theme_score'].max(),
            'theme_score_min': theme_df['theme_score'].min(),
            'research_score_avg': theme_df['research_score'].mean(),
            'societal_score_avg': theme_df['societal_score'].mean(),
            'unique_staff': theme_df['unique_staff'].sum(),
            'top_topic': theme_df.sort_values('theme_score', ascending=False).iloc[0]['theme_name'],
            'top_topic_score': theme_df['theme_score'].max(),
        }

        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)

    # Sort by average theme score
    summary_df = summary_df.sort_values('theme_score_avg', ascending=False)
    summary_df['rank'] = range(1, len(summary_df) + 1)

    # Save summary
    output_file = output_dir / 'parent_theme_summary.csv'
    summary_df.to_csv(output_file, index=False)
    logger.info(f"✅ Saved parent theme summary to: {output_file}")

    # Display summary
    logger.info(f"\n{'='*60}")
    logger.info("PARENT THEME COMPARISON")
    logger.info('='*60 + '\n')

    for _, row in summary_df.iterrows():
        logger.info(f"{row['rank']}. {row['parent_theme']}")
        logger.info(f"   Average Score: {row['theme_score_avg']:.1f} (range: {row['theme_score_min']:.1f}-{row['theme_score_max']:.1f})")
        logger.info(f"   Topics: {row['topics_count']} | Publications: {row['publications_total']} ({row['publications_scored']} scored)")
        logger.info(f"   Data Completeness: {row['data_completeness_rate']:.1f}%")
        logger.info(f"   Top Topic: {row['top_topic'][:60]} (score: {row['top_topic_score']:.1f})")
        logger.info("")

    return summary_df


def generate_detailed_report(df: pd.DataFrame, output_dir: Path):
    """Generate detailed markdown report."""

    report_file = output_dir / 'hierarchical_theme_report.md'

    with open(report_file, 'w') as f:
        f.write("# Hierarchical Theme Analysis Report\n\n")
        f.write(f"**Total Topics Analyzed:** {len(df)}\n")
        f.write(f"**Parent Themes:** {df['parent_theme'].nunique()}\n")
        f.write(f"**Total Publications:** {df['publications'].sum()}\n")
        f.write(f"**Overall Completeness:** {df['data_completeness_rate'].mean():.1f}%\n\n")

        f.write("---\n\n")

        # Section for each parent theme
        for parent in sorted(df['parent_theme'].unique()):
            theme_df = df[df['parent_theme'] == parent].sort_values('theme_score', ascending=False)

            f.write(f"## {parent}\n\n")

            # Theme-level summary
            f.write(f"**Topics:** {len(theme_df)}\n")
            f.write(f"**Total Publications:** {theme_df['publications'].sum()}\n")
            f.write(f"**Scored Publications:** {theme_df['publications_scored'].sum()}\n")
            f.write(f"**Average Score:** {theme_df['theme_score'].mean():.1f}\n")
            f.write(f"**Data Completeness:** {theme_df['data_completeness_rate'].mean():.1f}%\n\n")

            # Topic rankings table
            f.write("### Topic Rankings\n\n")
            f.write("| Rank | Topic | Score | Pubs | Scored | Excluded | Completeness | Research | Societal | Staff |\n")
            f.write("|------|-------|-------|------|--------|----------|--------------|----------|----------|-------|\n")

            for rank, (_, row) in enumerate(theme_df.iterrows(), 1):
                f.write(f"| {rank} | {row['theme_name']} | {row['theme_score']:.1f} | ")
                f.write(f"{row['publications']} | {row['publications_scored']} | {row['publications_excluded']} | ")
                f.write(f"{row['data_completeness_rate']:.1f}% | {row['research_score']:.1f} | ")
                f.write(f"{row['societal_score']:.1f} | {row['unique_staff']} |\n")

            f.write("\n")

            # Key insights
            f.write("### Key Insights\n\n")
            top_topic = theme_df.iloc[0]
            f.write(f"- **Highest Scoring Topic:** {top_topic['theme_name']} ({top_topic['theme_score']:.1f})\n")
            f.write(f"- **Most Publications:** {theme_df.loc[theme_df['publications'].idxmax(), 'theme_name']} ")
            f.write(f"({theme_df['publications'].max()} papers)\n")

            low_completeness = theme_df[theme_df['data_completeness_rate'] < 70]
            if len(low_completeness) > 0:
                f.write(f"- **⚠️ Low Completeness Topics:** {len(low_completeness)} topics below 70% completeness\n")

            f.write("\n---\n\n")

    logger.info(f"✅ Saved detailed report to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hierarchical theme results and rank topics within themes"
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to theme_comparison.csv from theme search'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for hierarchical analysis'
    )

    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info("HIERARCHICAL THEME ANALYSIS")
    logger.info("="*60 + "\n")

    try:
        # Load results
        df = load_theme_comparison(args.results)

        # Generate rankings within each theme
        rankings = generate_theme_rankings(df, args.output)

        # Generate parent theme summary
        summary = generate_theme_summary(df, args.output)

        # Generate detailed report
        generate_detailed_report(rankings, args.output)

        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info("\nGenerated files:")
        logger.info(f"  1. topic_rankings_by_theme.csv - Topics ranked within each parent theme")
        logger.info(f"  2. parent_theme_summary.csv - Summary statistics per parent theme")
        logger.info(f"  3. hierarchical_theme_report.md - Detailed markdown report")
        logger.info("\n")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
