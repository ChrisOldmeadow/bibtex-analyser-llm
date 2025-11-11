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

    def __init__(self, baselines: Optional[Dict] = None, global_semantic_threshold: Optional[float] = None):
        """Initialize with dataset baselines for normalization.

        Args:
            baselines: Optional dictionary with baseline metrics from calculate_dataset_baselines().
                      If None, will use OpenAlex data where available and mark missing as np.nan.
                      No imputation is performed.
            global_semantic_threshold: Optional global semantic cutoff (unused placeholder for compatibility).
        """
        self.baselines = baselines
        self.global_semantic_threshold = global_semantic_threshold
        self.output_cap = 250.0  # default cap
        self.normalized_cap = 2.0  # default FWCI cap

    def set_output_cap(self, publication_counts: List[int], percentile: float = 90.0) -> None:
        """Adjust the publication normalization cap based on percentile of counts."""
        valid = [count for count in publication_counts if count and count > 0]
        if not valid:
            self.output_cap = 250.0
            return
        cap = float(np.percentile(valid, percentile))
        if cap <= 0:
            cap = float(max(valid))
        self.output_cap = max(cap, 1.0)

    def set_normalized_cap(self, normalized_values: List[float], percentile: float = 90.0) -> None:
        """Adjust the normalized impact cap based on percentile of FWCI averages."""
        valid = [value for value in normalized_values if value is not None and not pd.isna(value) and value > 0]
        if not valid:
            self.normalized_cap = 2.0
            return
        cap = float(np.percentile(valid, percentile))
        if cap <= 0:
            cap = float(max(valid))
        self.normalized_cap = max(cap, 0.5)

    def preview_normalized_impact(self, theme_df: pd.DataFrame) -> float:
        """Return the raw average normalized impact for a theme (without scaling)."""
        return self._calculate_normalized_impact(theme_df)

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

        # 1. Output normalized to configured cap (default 250, or percentile-based)
        output_normalized = min(total_pubs / self.output_cap, 1.0) if self.output_cap > 0 else 0.0

        # 2. International Collaboration
        intl_collab_rate = self._calculate_intl_collaboration(theme_df)

        # 3. Q1 Publications
        q1_rate = self._calculate_q1_percentage(theme_df)

        # 4. Normalized Impact
        avg_normalized_impact = self._calculate_normalized_impact(theme_df)

        # 5. Excellence (top 10% most cited)
        excellence_rate = self._calculate_excellence_rate(theme_df, total_pubs)

        # 6. Leadership (first or last author)
        leadership_rate = self._calculate_leadership_rate(theme_df)

        # 7. Open Access
        oa_rate = self._calculate_open_access_rate(theme_df)

        def pct(value):
            if value is None or pd.isna(value):
                return np.nan
            return round(value * 100, 2)

        def rnd(value, digits):
            if value is None or pd.isna(value):
                return np.nan
            return round(value, digits)

        return {
            'output_normalized': rnd(output_normalized, 3),
            'intl_collaboration_rate': pct(intl_collab_rate),
            'q1_percentage': pct(q1_rate),
            'normalized_impact': rnd(avg_normalized_impact, 3),
            'excellence_rate': pct(excellence_rate),
            'leadership_rate': pct(leadership_rate),
            'open_access_rate': pct(oa_rate),
        }

    def _calculate_societal_components(self, theme_df: pd.DataFrame) -> Dict:
        """Calculate societal impact components.

        Uses np.nan for missing data - no imputation.
        """
        total_pubs = len(theme_df)

        # Get papers with altmetric scores - keep missing as missing, don't impute
        altmetric_col = None
        for col in ['Altmetrics_Score', 'altmetrics_score']:
            if col in theme_df.columns:
                altmetric_col = col
                break

        if altmetric_col:
            altmetric_numeric = pd.to_numeric(theme_df[altmetric_col], errors='coerce')
            altmetric_scores = altmetric_numeric.dropna()
            altmetric_scores = altmetric_scores[altmetric_scores > 0]
        else:
            altmetric_scores = pd.Series(dtype=float)

        # Coverage: % papers with any altmetric attention
        coverage_rate = len(altmetric_scores) / total_pubs if total_pubs > 0 else 0

        avg_altmetric = altmetric_scores.mean() if len(altmetric_scores) > 0 else np.nan

        return {
            'altmetric_coverage': round(coverage_rate * 100, 2),
            'avg_altmetric_score': avg_altmetric if pd.isna(avg_altmetric) else round(avg_altmetric, 2),
        }

    def _calculate_intl_collaboration(self, theme_df: pd.DataFrame) -> float:
        """Calculate international collaboration rate."""
        countries_col = None
        for col in ['Countries', 'countries', 'openalex_countries']:
            if col in theme_df.columns:
                countries_col = col
                break

        if not countries_col:
            return np.nan

        countries_series = theme_df[countries_col].dropna()
        if countries_series.empty:
            return np.nan

        intl_count = 0
        total_with_data = 0
        for countries_str in countries_series:
            countries = [c.strip() for c in str(countries_str).replace(';', ',').split(',') if c.strip()]
            if not countries:
                continue
            total_with_data += 1
            if len(countries) > 1:
                intl_count += 1

        if total_with_data == 0:
            return np.nan
        return intl_count / total_with_data

    def _calculate_q1_percentage(self, theme_df: pd.DataFrame) -> float:
        """Calculate percentage of Q1 publications."""
        q1_count = 0
        total_with_data = 0

        for _, row in theme_df.iterrows():
            quartile = row.get('Clarivate_Quartile_Rank')
            if pd.isna(quartile):
                quartile = row.get('SJR_Best_Quartile')
            if pd.isna(quartile):
                quartile = row.get('clarivate_quartile_rank')
            if pd.isna(quartile):
                quartile = row.get('sjr_best_quartile')

            if pd.isna(quartile) or str(quartile).strip() == '':
                continue

            total_with_data += 1
            if str(quartile).strip().upper() == 'Q1':
                q1_count += 1

        if total_with_data == 0:
            return np.nan
        return q1_count / total_with_data

    def _calculate_normalized_impact(self, theme_df: pd.DataFrame) -> float:
        """Calculate field-normalized citation impact.

        Prioritizes OpenAlex FWCI when available, falls back to baselines.
        Papers with no data are skipped (not imputed).

        Returns:
            Tuple of (average_normalized_impact, score_for_calculation)
        """
        normalized_impacts = []

        for _, row in theme_df.iterrows():
            # PRIORITY 1: Use OpenAlex FWCI if available (already field-normalized)
            openalex_fwci = row.get('openalex_fwci_approx')
            if pd.notna(openalex_fwci):
                try:
                    fwci_value = float(openalex_fwci)
                except (TypeError, ValueError):
                    fwci_value = None
                if fwci_value is not None and fwci_value > 0:
                    normalized_impacts.append(fwci_value)
                    continue

            # PRIORITY 2: Calculate from baselines (if available)
            if self.baselines:
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
                # If no expected baseline, skip this paper (don't impute)
            # If no OpenAlex data and no baselines - skip this paper (don't impute)

        if normalized_impacts:
            return float(np.mean(normalized_impacts))
        return np.nan

    def _calculate_excellence_rate(self, theme_df: pd.DataFrame, total: int) -> float:
        """Calculate percentage of papers in top 10% most cited.

        Prioritizes OpenAlex citation percentile when available, falls back to baselines.
        Papers with no data are skipped (not imputed).
        """
        excellence_count = 0
        papers_with_data = 0

        for _, row in theme_df.iterrows():
            # PRIORITY 1: Use OpenAlex citation percentile if available
            openalex_percentile = row.get('openalex_citation_percentile')
            if pd.notna(openalex_percentile):
                papers_with_data += 1
                # Top 10% = 90th percentile or above
                try:
                    percentile_value = float(openalex_percentile)
                except (TypeError, ValueError):
                    percentile_value = None
                if percentile_value is not None and percentile_value >= 90.0:
                    excellence_count += 1
                    continue

            # PRIORITY 2: Calculate from baselines (if available)
            if self.baselines:
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

                papers_with_data += 1
                if citations >= p90_threshold:
                    excellence_count += 1
            # If no OpenAlex data and no baselines - skip this paper (don't impute)

        return excellence_count / papers_with_data if papers_with_data > 0 else 0

    def _calculate_leadership_rate(self, theme_df: pd.DataFrame) -> float:
        """Calculate percentage of papers with first or last authorship."""
        leadership_count = 0
        total_with_data = 0

        for _, row in theme_df.iterrows():
            first = row.get('First_Author')
            if first is None:
                first = row.get('first_author')
            last = row.get('Last_Author')
            if last is None:
                last = row.get('last_author')

            if pd.isna(first) and pd.isna(last):
                continue

            total_with_data += 1
            true_values = {True, 1, '1', 'true', 'True', 'TRUE', 'Y', 'y'}
            if first in true_values or last in true_values:
                leadership_count += 1

        if total_with_data == 0:
            return np.nan
        return leadership_count / total_with_data

    def _calculate_open_access_rate(self, theme_df: pd.DataFrame) -> float:
        """Calculate percentage of open access publications."""
        oa_count = 0
        total_with_data = 0

        for _, row in theme_df.iterrows():
            pmc_id = row.get('Ref_PMC_ID')
            if pmc_id is None:
                pmc_id = row.get('ref_pmc_id')

            openalex_is_oa = row.get('openalex_is_oa')
            is_oa_flag = None

            if pd.notna(pmc_id) and str(pmc_id).strip():
                is_oa_flag = True
            elif openalex_is_oa is not None:
                if isinstance(openalex_is_oa, str):
                    is_oa_flag = openalex_is_oa.strip().lower() in {'true', '1', 'yes'}
                else:
                    is_oa_flag = bool(openalex_is_oa)

            if is_oa_flag is None:
                continue

            total_with_data += 1
            if is_oa_flag:
                oa_count += 1

        if total_with_data == 0:
            return np.nan
        return oa_count / total_with_data

    def _calculate_research_score(self, components: Dict) -> float:
        """Calculate overall research performance score."""
        normalized_component = components['normalized_impact']
        if pd.isna(normalized_component) or self.normalized_cap <= 0:
            normalized_component = 0.0
        else:
            normalized_component = min(normalized_component / self.normalized_cap, 1.0)

        return (
            0.25 * components['output_normalized'] +
            0.15 * (components['intl_collaboration_rate'] / 100) +
            0.20 * (components['q1_percentage'] / 100) +
            0.10 * normalized_component +
            0.15 * (components['excellence_rate'] / 100) +
            0.10 * (components['leadership_rate'] / 100) +
            0.05 * (components['open_access_rate'] / 100)
        ) * 100

    def _calculate_societal_score(self, components: Dict) -> float:
        """Calculate overall societal impact score."""
        return (components['altmetric_coverage'])  # already %

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
            'excellence_rate': 0.0,
            'leadership_rate': 0.0,
            'open_access_rate': 0.0,
            'altmetric_coverage': 0.0,
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
