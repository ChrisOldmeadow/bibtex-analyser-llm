"""
Theme Analysis Module

Separate module for strategic research theme analysis using:
- Semantic/hybrid search for theme matching
- SCImago-style institutional scoring
- Field-normalized metrics
- Staff aggregations per theme

This module is independent from the dashboard and provides CLI-based batch analysis.
"""

from .pipeline import ThemeSearchPipeline
from .scoring import calculate_theme_score, ThemeScorer
from .baselines import calculate_dataset_baselines, BaselineCalculator

__all__ = [
    'ThemeSearchPipeline',
    'calculate_theme_score',
    'ThemeScorer',
    'calculate_dataset_baselines',
    'BaselineCalculator',
]
