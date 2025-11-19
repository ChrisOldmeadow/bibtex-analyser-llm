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

    def __init__(
        self,
        df: pd.DataFrame,
        baselines: Optional[Dict] = None,
        api_key: Optional[str] = None,
        max_candidates: Optional[int] = None,
        semantic_only: bool = False,
        prompt_on_overflow: bool = False,
        ignore_max_limits: bool = False,
        global_semantic_threshold: Optional[float] = None,
        affiliate_index: Optional[Path] = None,
    ):
        """Initialize theme search pipeline.

        Args:
            df: DataFrame with institutional publications
            baselines: Optional baseline metrics for scoring. If None, will use OpenAlex data
                      where available and mark missing data as np.nan (no imputation).
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            max_candidates: Optional override for per-theme max candidate count
            semantic_only: Skip GPT rerank and rely on embeddings only
            prompt_on_overflow: Prompt when embedding matches exceed the candidate cap
            affiliate_index: Optional path to HMRI affiliate index CSV for linking staff names
        """
        self.df = df
        self.baselines = baselines
        self.scorer = ThemeScorer(baselines, global_semantic_threshold)
        self.max_candidates = max_candidates
        self._max_candidates_override = max_candidates
        self._has_candidate_override = max_candidates is not None
        self.semantic_only = semantic_only
        self.prompt_on_overflow = prompt_on_overflow
        self.ignore_max_limits = ignore_max_limits
        self.global_semantic_threshold = global_semantic_threshold
        self.affiliate_index = affiliate_index
        self._affiliate_lookup = None

        # Load affiliate index if provided
        if self.affiliate_index:
            self._load_affiliate_index()

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

    def _load_affiliate_index(self):
        """Load and prepare the affiliate index for staff name lookups."""
        try:
            index_df = pd.read_csv(self.affiliate_index)
            required_columns = {"NumberPlate", "Staff_First_Name", "Staff_Surname"}
            missing = required_columns - set(index_df.columns)
            if missing:
                logger.warning(
                    f"Affiliate index missing required columns: {', '.join(sorted(missing))}. "
                    f"Staff name linking will be skipped."
                )
                return

            index_df = index_df.copy()
            index_df["NumberPlate"] = index_df["NumberPlate"].astype(str).str.strip()

            # Include optional columns if available
            for col in ["Staff_Faculty", "Staff_School"]:
                if col not in index_df.columns:
                    index_df[col] = pd.NA

            # Create lookup dataframe
            self._affiliate_lookup = (
                index_df[
                    [
                        "NumberPlate",
                        "Staff_First_Name",
                        "Staff_Surname",
                        "Staff_Faculty",
                        "Staff_School",
                    ]
                ]
                .drop_duplicates(subset="NumberPlate")
            )
            logger.info(f"  Loaded affiliate index with {len(self._affiliate_lookup)} staff members")

        except Exception as e:
            logger.warning(f"Failed to load affiliate index: {e}. Staff name linking will be skipped.")
            self._affiliate_lookup = None

    def _link_staff_names(self, staff_df: pd.DataFrame) -> pd.DataFrame:
        """Add staff names and details from affiliate index to staff summary.

        Args:
            staff_df: Staff summary DataFrame with 'staff_id' column

        Returns:
            DataFrame with added name columns (or original if no index available)
        """
        if self._affiliate_lookup is None or staff_df.empty:
            return staff_df

        if "staff_id" not in staff_df.columns:
            logger.warning("Staff summary missing 'staff_id' column - cannot link names")
            return staff_df

        staff_df = staff_df.copy()
        staff_df["staff_id"] = staff_df["staff_id"].astype(str).str.strip()

        # Merge with affiliate data
        lookup = self._affiliate_lookup.rename(columns={"NumberPlate": "numberplate"})
        merged = staff_df.merge(
            lookup,
            how="left",
            left_on="staff_id",
            right_on="numberplate",
        )

        # Create full name column
        merged["staff_full_name"] = (
            merged["Staff_First_Name"].fillna("").str.strip()
            + " "
            + merged["Staff_Surname"].fillna("").str.strip()
        )
        merged["staff_full_name"] = merged["staff_full_name"].str.strip().replace({"": pd.NA})

        # Drop the temporary merge key
        merged = merged.drop(columns=["numberplate"])

        # Reorder columns to put names first
        column_order = [
            "staff_id",
            "staff_full_name",
            "Staff_First_Name",
            "Staff_Surname",
        ]
        if "Staff_Faculty" in merged.columns:
            column_order.append("Staff_Faculty")
        if "Staff_School" in merged.columns:
            column_order.append("Staff_School")

        # Add remaining columns
        column_order += [
            col
            for col in merged.columns
            if col not in {
                "staff_id",
                "staff_full_name",
                "Staff_First_Name",
                "Staff_Surname",
                "Staff_Faculty",
                "Staff_School",
            }
        ]
        merged = merged[column_order]

        return merged

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

        theme_semantic_only = theme.get('semantic_only', self.semantic_only)
        prompt_on_overflow = theme.get('prompt_on_overflow', False) or self.prompt_on_overflow
        min_relevance = theme.get('min_llm_relevance', 6.0)
        semantic_threshold = self.global_semantic_threshold if self.global_semantic_threshold is not None else theme.get('semantic_threshold', 0.5)

        # Candidate cap
        if self._has_candidate_override:
            candidate_limit = self._max_candidates_override
        elif self.ignore_max_limits:
            candidate_limit = None
        elif theme_semantic_only:
            candidate_limit = theme.get('semantic_max_candidates')
        else:
            candidate_limit = theme.get('max_candidates', 100)

        # Max results (semantic-only ignores generic max_results unless a semantic-specific cap is set)
        if self.ignore_max_limits:
            max_results = None
        elif theme_semantic_only:
            max_results = theme.get('semantic_max_results')
        else:
            max_results = theme.get('max_results', 50)

        # Run hybrid (or semantic-only) search with precomputed embeddings
        results_tuples = self.searcher.hybrid_search(
            query=theme['narrative'],
            df=self.df,
            threshold=semantic_threshold,
            max_embedding_candidates=candidate_limit,
            max_results=max_results,
            logger=logger,
            precomputed_embeddings=self.precomputed_embeddings,
            prompt_on_overflow=prompt_on_overflow,
            semantic_only=theme_semantic_only,
        )

        if not results_tuples:
            logger.warning(f"No results found for theme: {theme_name}")
            return pd.DataFrame()

        if theme_semantic_only:
            theme_df = self._build_semantic_only_results(
                results_tuples,
                theme,
                theme_id,
                theme_name,
                semantic_threshold=semantic_threshold,
            )
            threshold_label = semantic_threshold
        else:
            theme_df = self._build_hybrid_results(theme, theme_id, theme_name, min_relevance)
            threshold_label = min_relevance

        if theme_df.empty:
            return theme_df

        threshold_desc = "semantic" if theme_semantic_only else "LLM relevance"
        logger.info(f"Found {len(theme_df)} publications above {threshold_desc} threshold {threshold_label}")

        # Save theme papers
        output_dir.mkdir(parents=True, exist_ok=True)
        papers_path = output_dir / "papers.csv"
        theme_df.to_csv(papers_path, index=False)
        logger.info(f"Saved papers to {papers_path}")

        return theme_df

    def _build_hybrid_results(self, theme: Dict, theme_id: str, theme_name: str, min_relevance: float) -> pd.DataFrame:
        """Assemble theme DataFrame using LLM-enhanced results."""
        analyzed_papers = getattr(self.searcher, '_last_analyzed_papers', [])

        result_rows = []
        for analyzed_paper in analyzed_papers:
            llm_score = analyzed_paper.get('llm_relevance_score', 0)

            if llm_score < min_relevance:
                continue

            idx = analyzed_paper['original_index']
            paper = self.df.iloc[idx].to_dict()
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
        theme_df = theme_df.sort_values('llm_relevance_score', ascending=False)
        return theme_df

    def _build_semantic_only_results(
        self,
        results_tuples: List[tuple],
        theme: Dict,
        theme_id: str,
        theme_name: str,
        semantic_threshold: float,
    ) -> pd.DataFrame:
        """Build results DataFrame when running without LLM rerank."""
        rows: List[Dict] = []
        cutoff = float(semantic_threshold or 0.0)

        for idx, score in results_tuples:
            if score < cutoff:
                continue
            pseudo_relevance = round(float(score) * 10, 3)

            paper = self.df.iloc[idx].to_dict()
            paper.update({
                'theme_id': theme_id,
                'theme_name': theme_name,
                'embedding_score': score,
                'llm_relevance_score': pseudo_relevance,
                'llm_confidence': 0.0,
                'llm_reasoning': 'Semantic-only mode (LLM rerank disabled)',
                'llm_key_concepts': '',
                'hybrid_score': score,
            })
            rows.append(paper)

        if not rows:
            logger.warning(f"No semantic matches above threshold {cutoff} for theme: {theme_name}")
            return pd.DataFrame()

        theme_df = pd.DataFrame(rows)
        theme_df = theme_df.sort_values('llm_relevance_score', ascending=False)
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

        # Additional quality metrics per staff
        def _is_q1(row):
            quartile = row.get('Clarivate_Quartile_Rank')
            if pd.isna(quartile):
                quartile = row.get('SJR_Best_Quartile')
            if pd.isna(quartile):
                quartile = row.get('clarivate_quartile_rank')
            if pd.isna(quartile):
                quartile = row.get('sjr_best_quartile')
            if pd.isna(quartile):
                return 0
            return 1 if str(quartile).strip().upper() == 'Q1' else 0

        def _is_lead(row):
            first = row.get('First_Author')
            if first is None:
                first = row.get('first_author')
            last = row.get('Last_Author')
            if last is None:
                last = row.get('last_author')
            true_vals = {True, 1, '1', 'true', 'True', 'TRUE', 'Y', 'y'}
            return 1 if (first in true_vals) or (last in true_vals) else 0

        def _is_oa(row):
            pmc_id = row.get('Ref_PMC_ID')
            if pmc_id is None:
                pmc_id = row.get('ref_pmc_id')
            openalex_is_oa = row.get('openalex_is_oa')
            if pd.notna(pmc_id) and str(pmc_id).strip():
                return 1
            if openalex_is_oa is not None:
                if isinstance(openalex_is_oa, str):
                    if openalex_is_oa.strip().lower() in {'true', '1', 'yes', 'y'}:
                        return 1
                elif bool(openalex_is_oa):
                    return 1
            return 0

        altmetric_col = None
        for candidate in ['Altmetrics_Score', 'altmetrics_score']:
            if candidate in staff_papers.columns:
                altmetric_col = candidate
                break

        staff_metrics = staff_papers.copy()
        staff_metrics['_q1_flag'] = staff_metrics.apply(_is_q1, axis=1)
        staff_metrics['_lead_flag'] = staff_metrics.apply(_is_lead, axis=1)
        staff_metrics['_oa_flag'] = staff_metrics.apply(_is_oa, axis=1)
        if altmetric_col:
            staff_metrics['_alt_score'] = pd.to_numeric(staff_metrics[altmetric_col], errors='coerce')
            staff_metrics['_alt_flag'] = (staff_metrics['_alt_score'] > 0).astype(int)
        else:
            staff_metrics['_alt_score'] = np.nan
            staff_metrics['_alt_flag'] = np.nan
        if 'embedding_score' in staff_metrics.columns:
            staff_metrics['_embed'] = staff_metrics['embedding_score']
        else:
            staff_metrics['_embed'] = np.nan

        metrics = staff_metrics.groupby(group_col).agg({
            '_q1_flag': 'mean',
            '_lead_flag': 'mean',
            '_oa_flag': 'mean',
            '_alt_flag': 'mean',
            '_alt_score': 'mean',
            '_embed': 'mean'
        }).reset_index().rename(columns={
            '_q1_flag': 'q1_rate',
            '_lead_flag': 'leadership_rate',
            '_oa_flag': 'open_access_rate',
            '_alt_flag': 'altmetric_coverage',
            '_alt_score': 'avg_altmetric_score',
            '_embed': 'avg_semantic_score'
        })
        metrics = metrics.rename(columns={group_col: 'staff_id'})

        for col in ['q1_rate', 'leadership_rate', 'open_access_rate', 'altmetric_coverage']:
            metrics[col] = (metrics[col] * 100).round(2)
        metrics['avg_altmetric_score'] = metrics['avg_altmetric_score'].round(2)
        metrics['avg_semantic_score'] = metrics['avg_semantic_score'].round(3)

        staff_summary = staff_summary.merge(metrics, on='staff_id', how='left')

        # Count high-quality papers (LLM score >= 8)
        high_quality = staff_papers[staff_papers['llm_relevance_score'] >= 8.0]
        high_quality_counts = high_quality.groupby(group_col).size().reset_index(name='high_quality_papers')
        high_quality_counts = high_quality_counts.rename(columns={group_col: 'staff_id'})
        staff_summary = staff_summary.merge(high_quality_counts, on='staff_id', how='left')
        staff_summary['high_quality_papers'] = staff_summary['high_quality_papers'].fillna(0).astype(int)

        # Sort by paper count (primary) then average relevance
        staff_summary = staff_summary.sort_values(['paper_count', 'avg_relevance'], ascending=[False, False])
        staff_summary['rank'] = range(1, len(staff_summary) + 1)

        # Add theme metadata
        staff_summary['theme_id'] = theme['id']
        staff_summary['theme_name'] = theme['name']

        # Link staff names from affiliate index if available
        staff_summary = self._link_staff_names(staff_summary)

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

        theme_results = []
        publication_counts = []
        normalized_avgs = []
        top_staff_entries = []

        for theme in themes:
            theme_output_dir = output_dir / theme['id']
            theme_output_dir.mkdir(parents=True, exist_ok=True)

            theme_df = self.run_theme(theme, theme_output_dir)
            theme_results.append((theme, theme_df, theme_output_dir))
            publication_counts.append(len(theme_df))
            normalized_avgs.append(self.scorer.preview_normalized_impact(theme_df))

        self.scorer.set_output_cap(publication_counts)
        self.scorer.set_normalized_cap(normalized_avgs)

        all_theme_scores = []
        all_staff_summaries = []

        for theme, theme_df, theme_output_dir in theme_results:
            if theme_df.empty:
                continue

            theme_score = self.scorer.score_theme(theme_df, theme['id'], theme['name'])
            theme_score['parent_theme'] = theme.get('parent_theme')
            if 'embedding_score' in theme_df.columns:
                theme_score['avg_semantic_score'] = float(theme_df['embedding_score'].mean())
            else:
                theme_score['avg_semantic_score'] = np.nan
            all_theme_scores.append(theme_score)

            staff_df = self.aggregate_staff_for_theme(theme_df, theme, theme_output_dir)
            if not staff_df.empty:
                all_staff_summaries.append(staff_df)
                top_staff_entries.append(staff_df.head(20).copy())

        # Create theme comparison CSV
        if all_theme_scores:
            comparison_df = pd.DataFrame(all_theme_scores)
        if 'avg_semantic_score' in comparison_df.columns:
            comparison_df['avg_semantic_score'] = comparison_df['avg_semantic_score'].round(3)
            group_key = comparison_df['parent_theme'].fillna(comparison_df['theme_id'])
            comparison_df['rank_within_parent'] = (
                comparison_df.groupby(group_key)['theme_score']
                .rank(method='dense', ascending=False)
                .astype(int)
            )
            comparison_out = comparison_df.sort_values(['parent_theme', 'rank_within_parent'])
            comparison_path = output_dir / "theme_comparison.csv"
            comparison_out.to_csv(comparison_path, index=False)
            logger.info(f"\n{'='*60}")
            logger.info(f"RESULTS SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Saved theme comparison to {comparison_path}")
            logger.info(f"\nTop 5 themes by score:")
            for i, row in comparison_df.sort_values('theme_score', ascending=False).head(5).iterrows():
                logger.info(f"  {i+1}. {row['theme_name']}: {row['theme_score']:.1f} "
                          f"({row['publications']} pubs, R:{row['research_score']:.1f}, "
                          f"S:{row['societal_score']:.1f})")

        # Optionally combine all staff summaries
        if all_staff_summaries:
            combined_staff = pd.concat(all_staff_summaries, ignore_index=True)
            combined_staff_path = output_dir / "all_themes_staff.csv"
            combined_staff.to_csv(combined_staff_path, index=False)
            logger.info(f"Saved combined staff summaries to {combined_staff_path}")

        if top_staff_entries:
            top_staff = pd.concat(top_staff_entries, ignore_index=True)
            top_staff_path = output_dir / "top_staff_by_theme.csv"
            top_staff.to_csv(top_staff_path, index=False)
            logger.info(f"Saved top 20 staff per theme to {top_staff_path}")

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
