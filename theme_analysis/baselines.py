"""
Baseline calculation for field normalization in theme scoring.

Calculates expected values from full dataset for:
- Citation rates by Field of Research (FoR) + Year + Type
- Excellence thresholds (90th percentile citations)
- Altmetric score distributions
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaselineCalculator:
    """Calculate baseline metrics from institutional publication dataset."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with full publication dataset.

        Args:
            df: DataFrame with institutional publications
        """
        self.df = df
        self.baselines = {
            'citations': {},
            'excellence': {},
            'altmetrics': {},
            'metadata': {}
        }

    def calculate_all(self) -> Dict:
        """Calculate all baseline metrics.

        Returns:
            Dictionary with baseline metrics for normalization
        """
        logger.info(f"Calculating baselines from {len(self.df)} publications")

        self._calculate_citation_baselines()
        self._calculate_excellence_thresholds()
        self._calculate_altmetric_baselines()
        self._add_metadata()

        logger.info("Baseline calculation complete")
        return self.baselines

    def _calculate_citation_baselines(self):
        """Calculate expected citations by FoR + Year + Type."""
        logger.info("Calculating citation baselines...")

        # Get best citation count per paper
        self.df['best_citations'] = self.df[['Citations_Scopus', 'Citations_WoS']].apply(
            lambda x: max(x.fillna(0)), axis=1
        )

        # Parse FoR codes (take first code if multiple)
        def extract_first_for(for_codes):
            if pd.isna(for_codes):
                return None
            for_str = str(for_codes).strip()
            if not for_str:
                return None
            # Split by comma or semicolon and take first
            parts = for_str.replace(';', ',').split(',')
            return parts[0].strip() if parts else None

        self.df['primary_for'] = self.df['FoR_Codes_2020'].apply(extract_first_for)

        # Group by FoR + Year + Type and calculate median citations
        citation_counts = defaultdict(list)
        for _, row in self.df.iterrows():
            for_code = row['primary_for']
            year = row.get('Reported_Year')
            pub_type = row.get('Publication_Type', 'Article')
            citations = row['best_citations']

            if pd.notna(for_code) and pd.notna(year) and pd.notna(citations):
                key = (str(for_code), int(year), str(pub_type))
                citation_counts[key].append(citations)

        # Calculate medians for groups with sufficient data
        min_sample_size = 5
        for key, citations in citation_counts.items():
            if len(citations) >= min_sample_size:
                for_code, year, pub_type = key
                median_cites = np.median(citations)
                self.baselines['citations'][f"{for_code}|{year}|{pub_type}"] = median_cites

        # Overall median as fallback
        overall_citations = self.df['best_citations'].dropna()
        if len(overall_citations) > 0:
            self.baselines['citations']['overall_median'] = float(np.median(overall_citations))
        else:
            self.baselines['citations']['overall_median'] = 0.0

        logger.info(f"  Calculated baselines for {len(self.baselines['citations'])} FoR/Year/Type groups")
        logger.info(f"  Overall median citations: {self.baselines['citations']['overall_median']:.1f}")

    def _calculate_excellence_thresholds(self):
        """Calculate 90th percentile citation thresholds by FoR + Year."""
        logger.info("Calculating excellence thresholds...")

        # Group by FoR + Year and calculate 90th percentile
        excellence_counts = defaultdict(list)
        for _, row in self.df.iterrows():
            for_code = row['primary_for']
            year = row.get('Reported_Year')
            citations = row['best_citations']

            if pd.notna(for_code) and pd.notna(year) and pd.notna(citations):
                key = (str(for_code), int(year))
                excellence_counts[key].append(citations)

        # Calculate 90th percentiles for groups with sufficient data
        min_sample_size = 10
        for key, citations in excellence_counts.items():
            if len(citations) >= min_sample_size:
                for_code, year = key
                p90 = np.percentile(citations, 90)
                self.baselines['excellence'][f"{for_code}|{year}"] = p90

        # Overall 90th percentile as fallback
        overall_citations = self.df['best_citations'].dropna()
        if len(overall_citations) > 0:
            self.baselines['excellence']['overall_p90'] = float(np.percentile(overall_citations, 90))
        else:
            self.baselines['excellence']['overall_p90'] = 0.0

        logger.info(f"  Calculated thresholds for {len(self.baselines['excellence'])} FoR/Year groups")
        logger.info(f"  Overall 90th percentile: {self.baselines['excellence']['overall_p90']:.1f} citations")

    def _calculate_altmetric_baselines(self):
        """Calculate altmetric distribution statistics."""
        logger.info("Calculating altmetric baselines...")

        altmetric_scores = self.df['Altmetrics_Score'].dropna()
        altmetric_scores = altmetric_scores[altmetric_scores > 0]  # Only non-zero scores

        if len(altmetric_scores) > 0:
            self.baselines['altmetrics']['p95'] = float(np.percentile(altmetric_scores, 95))
            self.baselines['altmetrics']['mean'] = float(np.mean(altmetric_scores))
            self.baselines['altmetrics']['median'] = float(np.median(altmetric_scores))
            self.baselines['altmetrics']['coverage_rate'] = float(len(altmetric_scores) / len(self.df) * 100)

            logger.info(f"  95th percentile altmetric score: {self.baselines['altmetrics']['p95']:.1f}")
            logger.info(f"  Altmetric coverage: {self.baselines['altmetrics']['coverage_rate']:.1f}%")
        else:
            self.baselines['altmetrics']['p95'] = 100.0  # Default if no altmetric data
            self.baselines['altmetrics']['mean'] = 0.0
            self.baselines['altmetrics']['median'] = 0.0
            self.baselines['altmetrics']['coverage_rate'] = 0.0
            logger.warning("  No altmetric data found in dataset")

    def _add_metadata(self):
        """Add metadata about the baseline calculation."""
        from datetime import datetime

        self.baselines['metadata'] = {
            'calculation_date': datetime.now().isoformat(),
            'total_publications': len(self.df),
            'unique_for_codes': int(self.df['primary_for'].nunique()),
            'year_range': f"{int(self.df['Reported_Year'].min())}-{int(self.df['Reported_Year'].max())}",
            'has_scopus_citations': int((self.df['Citations_Scopus'] > 0).sum()),
            'has_wos_citations': int((self.df['Citations_WoS'] > 0).sum()),
            'has_altmetrics': int((self.df['Altmetrics_Score'] > 0).sum()),
        }

    def save(self, output_path: Path):
        """Save baselines to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.baselines, f, indent=2)

        logger.info(f"Baselines saved to {output_path}")

    @staticmethod
    def load(input_path: Path) -> Dict:
        """Load baselines from JSON file.

        Args:
            input_path: Path to baselines JSON file

        Returns:
            Dictionary with baseline metrics
        """
        with open(input_path, 'r') as f:
            baselines = json.load(f)

        logger.info(f"Loaded baselines from {input_path}")
        logger.info(f"  Dataset: {baselines['metadata']['total_publications']} publications")
        logger.info(f"  Years: {baselines['metadata']['year_range']}")

        return baselines


def calculate_dataset_baselines(df: pd.DataFrame, output_path: Optional[Path] = None) -> Dict:
    """Calculate baseline metrics from full dataset.

    Args:
        df: DataFrame with institutional publications
        output_path: Optional path to save baselines JSON

    Returns:
        Dictionary with baseline metrics
    """
    calculator = BaselineCalculator(df)
    baselines = calculator.calculate_all()

    if output_path:
        calculator.save(output_path)

    return baselines


def load_baselines(path: Path) -> Dict:
    """Load baselines from JSON file.

    Args:
        path: Path to baselines JSON

    Returns:
        Dictionary with baseline metrics
    """
    return BaselineCalculator.load(path)
