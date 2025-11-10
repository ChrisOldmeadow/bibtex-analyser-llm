"""
Theme search pipeline for strategic research analysis.

Orchestrates:
1. Hybrid semantic search per theme
2. LLM relevance filtering
3. SCImago-style scoring
4. Staff aggregations
5. Output generation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import pandas as pd
import numpy as np

from bibtex_analyzer.semantic_search import HybridSemanticSearcher
from .scoring import ThemeScorer
from .baselines import load_baselines

logger = logging.getLogger(__name__)


class ThemeSearchPipeline:
    """Execute batch searches across multiple strategic themes."""

    def __init__(self, df: pd.DataFrame, baselines: Dict, api_key: Optional[str] = None):
        """Initialize theme search pipeline.

        Args:
            df: DataFrame with institutional publications
            baselines: Baseline metrics for scoring
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        self.df = df
        self.baselines = baselines
        self.scorer = ThemeScorer(baselines)

        # Initialize hybrid searcher
        self.searcher = HybridSemanticSearcher(api_key=api_key)

        # Precompute embeddings for all papers (with caching)
        logger.info("Preparing dataset embeddings...")
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Precompute embeddings for all papers."""
        paper_texts = []
        for _, row in self.df.iterrows():
            text = self.searcher.prepare_paper_text(row.to_dict())
            paper_texts.append(text)

        # This will use cache where possible
        self.precomputed_embeddings = self.searcher.get_embeddings_batch(
            paper_texts,
            logger=logger
        )
        logger.info(f"  Embeddings ready for {len(self.precomputed_embeddings)} papers")

    def run_theme(self, theme: Dict, output_dir: Path) -> pd.DataFrame:
        """Execute hybrid search for a single theme.

        Args:
            theme: Theme definition with narrative, thresholds, etc.
            output_dir: Directory for this theme's outputs

        Returns:
            DataFrame with search results and theme metadata
        """
        theme_id = theme['id']
        theme_name = theme['name']

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing theme: {theme_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Narrative: {theme['narrative'][:150]}...")

        # Run hybrid search with precomputed embeddings
        results_tuples = self.searcher.hybrid_search(
            query=theme['narrative'],
            df=self.df,
            threshold=theme.get('semantic_threshold', 0.5),
            max_embedding_candidates=theme.get('max_candidates', 100),
            max_results=theme.get('max_results', 50),
            logger=logger,
            precomputed_embeddings=self.precomputed_embeddings
        )

        if not results_tuples:
            logger.warning(f"No results found for theme: {theme_name}")
            return pd.DataFrame()

        # Get analyzed papers from searcher
        analyzed_papers = self.searcher._last_analyzed_papers

        # Filter by minimum LLM relevance and build result DataFrame
        result_rows = []
        min_relevance = theme.get('min_llm_relevance', 6.0)

        for analyzed_paper in analyzed_papers:
            llm_score = analyzed_paper.get('llm_relevance_score', 0)

            if llm_score < min_relevance:
                continue

            # Get original paper data
            idx = analyzed_paper['original_index']
            paper = self.df.iloc[idx].to_dict()

            # Add search metadata
            paper.update({
                'theme_id': theme_id,
                'theme_name': theme_name,
                'embedding_score': analyzed_paper.get('embedding_score', 0),
                'llm_relevance_score': llm_score,
                'llm_confidence': analyzed_paper.get('llm_confidence', 0),
                'llm_reasoning': analyzed_paper.get('llm_reasoning', ''),
                'llm_key_concepts': ', '.join(analyzed_paper.get('llm_key_concepts', [])),
                'hybrid_score': analyzed_paper.get('hybrid_score', 0)
            })

            result_rows.append(paper)

        if not result_rows:
            logger.warning(f"No papers above relevance threshold {min_relevance} for theme: {theme_name}")
            return pd.DataFrame()

        theme_df = pd.DataFrame(result_rows)

        # Sort by LLM relevance score
        theme_df = theme_df.sort_values('llm_relevance_score', ascending=False)

        logger.info(f"Found {len(theme_df)} publications above relevance threshold {min_relevance}")

        # Save theme papers
        output_dir.mkdir(parents=True, exist_ok=True)
        papers_path = output_dir / "papers.csv"
        theme_df.to_csv(papers_path, index=False)
        logger.info(f"Saved papers to {papers_path}")

        return theme_df

    def aggregate_staff_for_theme(self, theme_df: pd.DataFrame, theme: Dict,
                                   output_dir: Path) -> pd.DataFrame:
        """Aggregate papers by staff member for a single theme.

        Args:
            theme_df: Papers for this theme
            theme: Theme definition
            output_dir: Output directory

        Returns:
            DataFrame with staff-level aggregates
        """
        if theme_df.empty:
            logger.warning(f"No papers to aggregate for theme: {theme['name']}")
            return pd.DataFrame()

        # Check if staff ID columns exist
        staff_col = None
        for col in ['all_staff_ids', 'NumberPlate', 'staff_id']:
            if col in theme_df.columns:
                staff_col = col
                break

        if not staff_col:
            logger.warning(f"No staff ID column found for theme: {theme['name']}")
            return pd.DataFrame()

        # Explode staff IDs if needed
        if staff_col == 'all_staff_ids':
            # Handle array column
            staff_papers = theme_df.copy()
            staff_papers['all_staff_ids'] = staff_papers['all_staff_ids'].apply(
                lambda x: x if isinstance(x, list) else [x] if x else []
            )
            staff_papers = staff_papers.explode('all_staff_ids')
            staff_papers = staff_papers[staff_papers['all_staff_ids'].notna()]
            group_col = 'all_staff_ids'
        else:
            # Single staff ID per row
            staff_papers = theme_df[theme_df[staff_col].notna()].copy()
            group_col = staff_col

        if staff_papers.empty:
            logger.warning(f"No staff IDs found for theme: {theme['name']}")
            return pd.DataFrame()

        # Aggregate metrics per staff member
        id_col = 'Publication_ID' if 'Publication_ID' in staff_papers.columns else 'ID'

        agg_dict = {
            id_col: 'count',
            'llm_relevance_score': ['mean', 'max', 'std'],
            'hybrid_score': ['mean', 'max']
        }

        staff_summary = staff_papers.groupby(group_col).agg(agg_dict).reset_index()

        # Flatten column names
        staff_summary.columns = [
            'staff_id',
            'paper_count',
            'avg_relevance', 'max_relevance', 'std_relevance',
            'avg_hybrid_score', 'max_hybrid_score'
        ]

        # Count high-quality papers (LLM score >= 8)
        high_quality = staff_papers[staff_papers['llm_relevance_score'] >= 8.0]
        high_quality_counts = high_quality.groupby(group_col).size().reset_index(name='high_quality_papers')
        staff_summary = staff_summary.merge(high_quality_counts, on='staff_id', how='left')
        staff_summary['high_quality_papers'] = staff_summary['high_quality_papers'].fillna(0).astype(int)

        # Sort by average relevance and paper count
        staff_summary = staff_summary.sort_values(['avg_relevance', 'paper_count'], ascending=False)

        # Add theme metadata
        staff_summary['theme_id'] = theme['id']
        staff_summary['theme_name'] = theme['name']

        # Save staff summary
        staff_summary_path = output_dir / "staff_summary.csv"
        staff_summary.to_csv(staff_summary_path, index=False)
        logger.info(f"Saved staff summary for {len(staff_summary)} researchers to {staff_summary_path}")

        return staff_summary

    def run_all_themes(self, themes_path: Path, output_dir: Path) -> Dict:
        """Run all themes and generate comparative analysis.

        Args:
            themes_path: Path to themes.yaml
            output_dir: Directory for all outputs

        Returns:
            Dictionary with theme statistics and output paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load themes configuration
        with open(themes_path, 'r') as f:
            config = yaml.safe_load(f)
        themes = config['themes']

        logger.info(f"\n{'='*60}")
        logger.info(f"THEME SEARCH PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"Dataset: {len(self.df)} publications")
        logger.info(f"Themes to process: {len(themes)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'='*60}\n")

        all_theme_scores = []
        all_staff_summaries = []

        for theme in themes:
            theme_output_dir = output_dir / theme['id']

            # Run search for this theme
            theme_df = self.run_theme(theme, theme_output_dir)

            if theme_df.empty:
                continue

            # Calculate theme score
            theme_score = self.scorer.score_theme(theme_df, theme['id'], theme['name'])
            all_theme_scores.append(theme_score)

            # Generate staff summary
            staff_df = self.aggregate_staff_for_theme(theme_df, theme, theme_output_dir)

            if not staff_df.empty:
                all_staff_summaries.append(staff_df)

        # Create theme comparison CSV
        if all_theme_scores:
            comparison_df = pd.DataFrame(all_theme_scores)
            comparison_df = comparison_df.sort_values('theme_score', ascending=False)
            comparison_path = output_dir / "theme_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"\n{'='*60}")
            logger.info(f"RESULTS SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Saved theme comparison to {comparison_path}")
            logger.info(f"\nTop 5 themes by score:")
            for i, row in comparison_df.head(5).iterrows():
                logger.info(f"  {i+1}. {row['theme_name']}: {row['theme_score']:.1f} "
                          f"({row['publications']} pubs, R:{row['research_score']:.1f}, "
                          f"S:{row['societal_score']:.1f})")

        # Optionally combine all staff summaries
        if all_staff_summaries:
            combined_staff = pd.concat(all_staff_summaries, ignore_index=True)
            combined_staff_path = output_dir / "all_themes_staff.csv"
            combined_staff.to_csv(combined_staff_path, index=False)
            logger.info(f"Saved combined staff summaries to {combined_staff_path}")

        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*60}\n")

        return {
            'themes_processed': len(all_theme_scores),
            'output_dir': str(output_dir),
            'comparison_file': str(comparison_path) if all_theme_scores else None
        }


def load_themes(themes_path: Path) -> List[Dict]:
    """Load and validate theme definitions.

    Args:
        themes_path: Path to themes.yaml

    Returns:
        List of theme dictionaries
    """
    with open(themes_path, 'r') as f:
        config = yaml.safe_load(f)

    themes = config.get('themes', [])

    # Validate required fields
    required_fields = ['id', 'name', 'narrative']
    for theme in themes:
        for field in required_fields:
            if field not in theme:
                raise ValueError(f"Theme missing required field '{field}': {theme}")

    return themes
