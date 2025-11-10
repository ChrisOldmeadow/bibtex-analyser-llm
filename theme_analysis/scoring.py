"""
SCImago-style institutional scoring for research themes.

Implements two-stage scoring methodology:
1. Research Performance Score (65%): Output, collaboration, quality, impact, leadership
2. Societal Impact Score (35%): Altmetric coverage and intensity

See docs/theme_scoring_methodology.md for complete algorithm details.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ThemeScorer:
    """Calculate SCImago-style theme scores."""

    def __init__(self, baselines: Dict):
        """Initialize with dataset baselines for normalization.

        Args:
            baselines: Dictionary with baseline metrics from calculate_dataset_baselines()
        """
        self.baselines = baselines

    def score_theme(self, theme_df: pd.DataFrame, theme_id: str, theme_name: str) -> Dict:
        """Calculate comprehensive theme score.

        Args:
            theme_df: DataFrame of publications for this theme
            theme_id: Theme identifier
            theme_name: Theme display name

        Returns:
            Dictionary with scores and component metrics
        """
        if theme_df.empty:
            logger.warning(f"Theme '{theme_name}' has no publications")
            return self._empty_score(theme_id, theme_name)

        total_pubs = len(theme_df)
        logger.info(f"Scoring theme '{theme_name}' with {total_pubs} publications")

        # Calculate research performance components
        research_components = self._calculate_research_components(theme_df)

        # Calculate societal impact components
        societal_components = self._calculate_societal_components(theme_df)

        # Combine into final scores
        research_score = self._calculate_research_score(research_components)
        societal_score = self._calculate_societal_score(societal_components)
        theme_score = 0.65 * research_score + 0.35 * societal_score

        logger.info(f"  Theme score: {theme_score:.1f} (R:{research_score:.1f}, S:{societal_score:.1f})")

        return {
            'theme_id': theme_id,
            'theme_name': theme_name,
            'publications': total_pubs,
            'theme_score': round(theme_score, 2),
            'research_score': round(research_score, 2),
            'societal_score': round(societal_score, 2),
            **research_components,
            **societal_components
        }

    def _calculate_research_components(self, theme_df: pd.DataFrame) -> Dict:
        """Calculate all research performance components."""
        total_pubs = len(theme_df)

        # 1. Output (normalized to 100 publications = max)
        output_normalized = min(total_pubs / 100.0, 1.0)

        # 2. International Collaboration
        intl_collab_rate = self._calculate_intl_collaboration(theme_df, total_pubs)

        # 3. Q1 Publications
        q1_rate = self._calculate_q1_percentage(theme_df, total_pubs)

        # 4. Normalized Impact
        avg_normalized_impact, normalized_impact_score = self._calculate_normalized_impact(theme_df)

        # 5. Excellence (top 10% most cited)
        excellence_rate = self._calculate_excellence_rate(theme_df, total_pubs)

        # 6. Leadership (first or last author)
        leadership_rate = self._calculate_leadership_rate(theme_df, total_pubs)

        # 7. Open Access
        oa_rate = self._calculate_open_access_rate(theme_df, total_pubs)

        return {
            'output_normalized': round(output_normalized, 3),
            'intl_collaboration_rate': round(intl_collab_rate * 100, 2),
            'q1_percentage': round(q1_rate * 100, 2),
            'normalized_impact': round(avg_normalized_impact, 3),
            'normalized_impact_score': round(normalized_impact_score, 3),
            'excellence_rate': round(excellence_rate * 100, 2),
            'leadership_rate': round(leadership_rate * 100, 2),
            'open_access_rate': round(oa_rate * 100, 2),
        }

    def _calculate_societal_components(self, theme_df: pd.DataFrame) -> Dict:
        """Calculate societal impact components."""
        total_pubs = len(theme_df)

        # Get papers with altmetric scores
        altmetric_scores = theme_df['Altmetrics_Score'].fillna(0)
        altmetric_scores = altmetric_scores[altmetric_scores > 0]

        # Coverage: % papers with any altmetric attention
        coverage_rate = len(altmetric_scores) / total_pubs if total_pubs > 0 else 0

        # Intensity: average score relative to 95th percentile
        if len(altmetric_scores) > 0:
            avg_altmetric = altmetric_scores.mean()
            p95_threshold = self.baselines['altmetrics'].get('p95', 100.0)
            intensity_rate = min(avg_altmetric / p95_threshold, 1.0) if p95_threshold > 0 else 0
        else:
            avg_altmetric = 0.0
            intensity_rate = 0.0

        return {
            'altmetric_coverage': round(coverage_rate * 100, 2),
            'altmetric_intensity': round(intensity_rate, 3),
            'avg_altmetric_score': round(avg_altmetric, 2),
        }

    def _calculate_intl_collaboration(self, theme_df: pd.DataFrame, total: int) -> float:
        """Calculate international collaboration rate."""
        intl_count = 0
        for countries_str in theme_df['Countries'].fillna(''):
            # Split by semicolon or comma
            countries = [c.strip() for c in str(countries_str).replace(';', ',').split(',') if c.strip()]
            if len(countries) > 1:  # Multiple countries = international
                intl_count += 1

        return intl_count / total if total > 0 else 0

    def _calculate_q1_percentage(self, theme_df: pd.DataFrame, total: int) -> float:
        """Calculate percentage of Q1 publications."""
        q1_count = 0

        for _, row in theme_df.iterrows():
            # Check Clarivate first, then SJR
            quartile = row.get('Clarivate_Quartile_Rank')
            if pd.isna(quartile):
                quartile = row.get('SJR_Best_Quartile')

            if str(quartile).strip().upper() == 'Q1':
                q1_count += 1

        return q1_count / total if total > 0 else 0

    def _calculate_normalized_impact(self, theme_df: pd.DataFrame) -> tuple:
        """Calculate field-normalized citation impact.

        Returns:
            Tuple of (average_normalized_impact, score_for_calculation)
        """
        normalized_impacts = []

        for _, row in theme_df.iterrows():
            # Get best citation count
            scopus = row.get('Citations_Scopus', 0) or 0
            wos = row.get('Citations_WoS', 0) or 0
            citations = max(float(scopus), float(wos))

            # Get expected citations for this field/year/type
            for_codes_str = row.get('FoR_Codes_2020', '')
            for_code = str(for_codes_str).replace(';', ',').split(',')[0].strip() if for_codes_str else None

            year = row.get('Reported_Year')
            pub_type = row.get('Publication_Type', 'Article')

            # Look up expected citations
            if for_code and year:
                key = f"{for_code}|{int(year)}|{pub_type}"
                expected = self.baselines['citations'].get(
                    key,
                    self.baselines['citations'].get('overall_median', 1.0)
                )
            else:
                expected = self.baselines['citations'].get('overall_median', 1.0)

            # Calculate normalized impact
            if expected > 0:
                normalized_impacts.append(citations / expected)
            else:
                normalized_impacts.append(1.0)  # Neutral if no baseline

        avg_normalized = np.mean(normalized_impacts) if normalized_impacts else 1.0

        # Cap at 2.0 for scoring (2.0 = twice world average = max score)
        score = min(avg_normalized / 2.0, 1.0)

        return avg_normalized, score

    def _calculate_excellence_rate(self, theme_df: pd.DataFrame, total: int) -> float:
        """Calculate percentage of papers in top 10% most cited."""
        excellence_count = 0

        for _, row in theme_df.iterrows():
            # Get best citation count
            scopus = row.get('Citations_Scopus', 0) or 0
            wos = row.get('Citations_WoS', 0) or 0
            citations = max(float(scopus), float(wos))

            # Get 90th percentile threshold for this field/year
            for_codes_str = row.get('FoR_Codes_2020', '')
            for_code = str(for_codes_str).replace(';', ',').split(',')[0].strip() if for_codes_str else None
            year = row.get('Reported_Year')

            if for_code and year:
                key = f"{for_code}|{int(year)}"
                p90_threshold = self.baselines['excellence'].get(
                    key,
                    self.baselines['excellence'].get('overall_p90', 0)
                )
            else:
                p90_threshold = self.baselines['excellence'].get('overall_p90', 0)

            if citations >= p90_threshold:
                excellence_count += 1

        return excellence_count / total if total > 0 else 0

    def _calculate_leadership_rate(self, theme_df: pd.DataFrame, total: int) -> float:
        """Calculate percentage of papers with first or last authorship."""
        leadership_count = 0

        for _, row in theme_df.iterrows():
            first = row.get('First_Author')
            last = row.get('Last_Author')

            # Handle various boolean representations
            if first in [True, 1, '1', 'true', 'True', 'TRUE'] or \
               last in [True, 1, '1', 'true', 'True', 'TRUE']:
                leadership_count += 1

        return leadership_count / total if total > 0 else 0

    def _calculate_open_access_rate(self, theme_df: pd.DataFrame, total: int) -> float:
        """Calculate percentage of open access publications."""
        oa_count = 0

        for _, row in theme_df.iterrows():
            # Check for PMC ID (definitive OA indicator)
            pmc_id = row.get('Ref_PMC_ID')
            if pd.notna(pmc_id) and str(pmc_id).strip():
                oa_count += 1
                continue

            # TODO: Could add additional OA detection:
            # - Check DOI prefix (e.g., 10.1371 = PLOS)
            # - Check URL for repository indicators
            # - Check publisher field for known OA publishers

        return oa_count / total if total > 0 else 0

    def _calculate_research_score(self, components: Dict) -> float:
        """Calculate overall research performance score."""
        return (
            0.15 * components['output_normalized'] +
            0.15 * (components['intl_collaboration_rate'] / 100) +
            0.20 * (components['q1_percentage'] / 100) +
            0.20 * components['normalized_impact_score'] +
            0.15 * (components['excellence_rate'] / 100) +
            0.10 * (components['leadership_rate'] / 100) +
            0.05 * (components['open_access_rate'] / 100)
        ) * 100

    def _calculate_societal_score(self, components: Dict) -> float:
        """Calculate overall societal impact score."""
        return (
            0.60 * (components['altmetric_coverage'] / 100) +
            0.40 * components['altmetric_intensity']
        ) * 100

    def _empty_score(self, theme_id: str, theme_name: str) -> Dict:
        """Return zero scores for empty theme."""
        return {
            'theme_id': theme_id,
            'theme_name': theme_name,
            'publications': 0,
            'theme_score': 0.0,
            'research_score': 0.0,
            'societal_score': 0.0,
            'output_normalized': 0.0,
            'intl_collaboration_rate': 0.0,
            'q1_percentage': 0.0,
            'normalized_impact': 0.0,
            'normalized_impact_score': 0.0,
            'excellence_rate': 0.0,
            'leadership_rate': 0.0,
            'open_access_rate': 0.0,
            'altmetric_coverage': 0.0,
            'altmetric_intensity': 0.0,
            'avg_altmetric_score': 0.0,
        }


def calculate_theme_score(theme_df: pd.DataFrame, baselines: Dict,
                         theme_id: str = '', theme_name: str = '') -> Dict:
    """Calculate SCImago-style theme score.

    Args:
        theme_df: DataFrame of publications for this theme
        baselines: Baseline metrics from calculate_dataset_baselines()
        theme_id: Theme identifier
        theme_name: Theme display name

    Returns:
        Dictionary with scores and metrics
    """
    scorer = ThemeScorer(baselines)
    return scorer.score_theme(theme_df, theme_id, theme_name)
