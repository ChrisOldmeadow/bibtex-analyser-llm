"""Staff analysis module for ranking researchers based on publication frequency and quality."""

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict


class StaffAnalyzer:
    """Analyzes staff performance based on search results with intelligent handling of missing data."""
    
    def __init__(self, current_year: Optional[int] = None):
        """Initialize the staff analyzer.
        
        Args:
            current_year: Current year for citation normalization (defaults to current year)
        """
        self.current_year = current_year or datetime.now().year
        
        # Default weights for different metrics
        self.default_weights = {
            'relevance': 0.4,
            'citations': 0.3,
            'journal': 0.2,
            'altmetrics': 0.1
        }
        
        # Quartile to score mapping
        self.quartile_scores = {
            'Q1': 1.0,
            'Q2': 0.75,
            'Q3': 0.5,
            'Q4': 0.25,
            'unknown': 0.5  # Default for missing data
        }
    
    def calculate_quality_score(self, paper: Dict[str, Any], 
                              weights: Optional[Dict[str, float]] = None) -> Tuple[float, float, Dict[str, float]]:
        """Calculate quality score for a paper with adaptive weighting based on available data.
        
        Args:
            paper: Dictionary containing paper data
            weights: Optional custom weights for metrics
            
        Returns:
            Tuple of (quality_score, data_completeness, metrics_used)
        """
        weights = weights or self.default_weights.copy()
        metrics = {}
        available_metrics = []
        
        # 1. Search relevance (always available)
        relevance_score = self._get_relevance_score(paper)
        if relevance_score is not None:
            metrics['relevance'] = relevance_score
            available_metrics.append('relevance')
        
        # 2. Citation score (normalized by time)
        citation_score = self._get_citation_score(paper)
        if citation_score is not None:
            metrics['citations'] = citation_score
            available_metrics.append('citations')
        
        # 3. Journal quality score
        journal_score = self._get_journal_score(paper)
        if journal_score is not None:
            metrics['journal'] = journal_score
            available_metrics.append('journal')
        
        # 4. Altmetrics score
        altmetrics_score = self._get_altmetrics_score(paper)
        if altmetrics_score is not None:
            metrics['altmetrics'] = altmetrics_score
            available_metrics.append('altmetrics')
        
        # Calculate weighted score with available metrics
        if not metrics:
            return 0.0, 0.0, {}
        
        # Adjust weights based on available metrics
        available_weights = {k: weights[k] for k in available_metrics}
        total_weight = sum(available_weights.values())
        
        if total_weight == 0:
            return 0.0, 0.0, metrics
        
        # Normalize weights to sum to 1
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
        
        # Calculate weighted score
        quality_score = sum(metrics[k] * normalized_weights[k] for k in metrics)
        
        # Calculate data completeness
        data_completeness = len(metrics) / len(self.default_weights)
        
        # Apply confidence adjustment based on data completeness
        confidence_adjusted_score = quality_score * (0.7 + 0.3 * data_completeness)
        
        return confidence_adjusted_score, data_completeness, metrics
    
    def _get_relevance_score(self, paper: Dict) -> Optional[float]:
        """Extract and normalize relevance score from search results."""
        # Try different possible field names
        for field in ['search_score', 'relevance_score', 'llm_relevance_score']:
            if field in paper and paper[field] is not None:
                # Normalize to 0-1 range
                if field == 'llm_relevance_score':
                    return paper[field] / 10.0  # LLM scores are 0-10
                return float(paper[field])
        return None
    
    def _get_citation_score(self, paper: Dict) -> Optional[float]:
        """Calculate normalized citation score considering publication year."""
        # Get total citations from all available sources
        citation_fields = [
            'citations_europe_pubmed', 'citations_scopus', 
            'citations_wos', 'citations_wos_lite', 'citations_scival'
        ]
        
        citations = []
        for field in citation_fields:
            value = paper.get(field)
            if value is not None and str(value).strip() and str(value).lower() != 'nan':
                try:
                    citations.append(float(value))
                except (ValueError, TypeError):
                    pass
        
        if not citations:
            return None
        
        # Use maximum citation count from all sources
        total_citations = max(citations)
        
        # Get publication year
        year = self._extract_year(paper)
        if year is None:
            # If no year, use raw citation count normalized by 100
            return min(total_citations / 100.0, 1.0)
        
        # Calculate years since publication
        years_since_pub = max(self.current_year - year, 1)
        
        # Normalize based on age
        if years_since_pub <= 2:
            # Recent papers: expect fewer citations
            normalized_score = min(total_citations / 10.0, 1.0)
        elif years_since_pub <= 5:
            # Mid-age papers
            normalized_score = min(total_citations / 50.0, 1.0)
        else:
            # Older papers: citations per year
            citations_per_year = total_citations / years_since_pub
            normalized_score = min(citations_per_year / 10.0, 1.0)
        
        return normalized_score
    
    def _get_journal_score(self, paper: Dict) -> Optional[float]:
        """Extract journal quality score from quartile rankings."""
        # Check Clarivate quartile first
        clarivate = paper.get('clarivate_quartile_rank')
        if clarivate and str(clarivate).strip() and str(clarivate).lower() != 'nan':
            quartile = str(clarivate).upper()
            if quartile in self.quartile_scores:
                return self.quartile_scores[quartile]
        
        # Check SJR quartile
        sjr = paper.get('sjr_best_quartile')
        if sjr and str(sjr).strip() and str(sjr).lower() != 'nan':
            quartile = str(sjr).upper()
            if quartile in self.quartile_scores:
                return self.quartile_scores[quartile]
        
        return None
    
    def _get_altmetrics_score(self, paper: Dict) -> Optional[float]:
        """Extract and normalize altmetrics score."""
        altmetrics = paper.get('altmetrics_score')
        if altmetrics is not None and str(altmetrics).strip() and str(altmetrics).lower() != 'nan':
            try:
                score = float(altmetrics)
                # Normalize altmetrics (typically 0-100+ range)
                return min(score / 50.0, 1.0)
            except (ValueError, TypeError):
                pass
        
        # Don't penalize old papers for missing altmetrics
        year = self._extract_year(paper)
        if year and year < 2010:
            return None  # Not expected for old papers
        
        return None
    
    def _extract_year(self, paper: Dict) -> Optional[int]:
        """Extract publication year from paper data."""
        year_field = paper.get('year') or paper.get('reported_year')
        if year_field:
            try:
                year = int(float(str(year_field)))
                if 1900 <= year <= self.current_year:
                    return year
            except (ValueError, TypeError):
                pass
        return None
    
    def aggregate_staff_metrics(self, search_results: pd.DataFrame, 
                              min_completeness: float = 0.0) -> pd.DataFrame:
        """Aggregate search results by staff and calculate composite metrics.
        
        Args:
            search_results: DataFrame of search results
            min_completeness: Minimum data completeness threshold (0-1)
            
        Returns:
            DataFrame with staff metrics and rankings
        """
        staff_data = defaultdict(lambda: {
            'publications': [],
            'total_count': 0,
            'relevance_scores': [],
            'quality_scores': [],
            'completeness_scores': [],
            'total_citations': 0,
            'years': []
        })
        
        # Process each publication
        for idx, row in search_results.iterrows():
            paper = row.to_dict()
            
            # Get all staff IDs for this publication
            staff_ids = []
            if 'all_staff_ids' in paper and paper['all_staff_ids']:
                if isinstance(paper['all_staff_ids'], list):
                    staff_ids.extend(paper['all_staff_ids'])
                else:
                    # Handle comma-separated string
                    staff_ids.extend([s.strip() for s in str(paper['all_staff_ids']).split(',')])
            
            # Also check for single staff_id (might be in addition to all_staff_ids)
            if 'staff_id' in paper and paper['staff_id']:
                staff_id = str(paper['staff_id']).strip()
                if staff_id not in staff_ids:
                    staff_ids.append(staff_id)
            
            # Skip if no staff identified
            if not staff_ids:
                continue
            
            # Filter out empty or invalid staff IDs
            staff_ids = [sid for sid in staff_ids if sid and str(sid).strip() and str(sid).strip().lower() != 'nan']
            if not staff_ids:
                continue
            
            # Calculate quality metrics for this paper
            quality_score, completeness, metrics = self.calculate_quality_score(paper)
            
            # Skip if below completeness threshold
            if completeness < min_completeness:
                continue
            
            # Get citation count
            citations = self._get_total_citations(paper)
            
            # Add to each staff member's record
            for staff_id in staff_ids:
                if staff_id and str(staff_id).strip():
                    data = staff_data[staff_id]
                    data['publications'].append(paper)
                    data['total_count'] += 1
                    
                    # Add relevance score
                    relevance = self._get_relevance_score(paper)
                    if relevance is not None:
                        data['relevance_scores'].append(relevance)
                    
                    # Add quality metrics
                    data['quality_scores'].append(quality_score)
                    data['completeness_scores'].append(completeness)
                    data['total_citations'] += citations
                    
                    # Add year
                    year = self._extract_year(paper)
                    if year:
                        data['years'].append(year)
        
        # Create summary DataFrame
        summary_data = []
        for staff_id, data in staff_data.items():
            if data['total_count'] > 0:
                summary = {
                    'staff_id': staff_id,
                    'publication_count': data['total_count'],
                    'avg_relevance': np.mean(data['relevance_scores']) if data['relevance_scores'] else 0,
                    'avg_quality': np.mean(data['quality_scores']) if data['quality_scores'] else 0,
                    'avg_completeness': np.mean(data['completeness_scores']) if data['completeness_scores'] else 0,
                    'total_citations': data['total_citations'],
                    'citations_per_pub': data['total_citations'] / data['total_count'],
                    'year_range': f"{min(data['years'])}-{max(data['years'])}" if data['years'] else "N/A",
                    'active_years': len(set(data['years'])) if data['years'] else 0
                }
                
                # Calculate composite impact score
                summary['impact_score'] = self._calculate_impact_score(summary)
                
                summary_data.append(summary)
        
        # Create DataFrame and sort by impact score
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values('impact_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _get_total_citations(self, paper: Dict) -> float:
        """Get total citation count from all sources."""
        citation_fields = [
            'citations_europe_pubmed', 'citations_scopus', 
            'citations_wos', 'citations_wos_lite', 'citations_scival'
        ]
        
        total = 0
        for field in citation_fields:
            value = paper.get(field)
            if value is not None and str(value).strip() and str(value).lower() != 'nan':
                try:
                    total += float(value)
                except (ValueError, TypeError):
                    pass
        
        return total
    
    def _calculate_impact_score(self, staff_summary: Dict) -> float:
        """Calculate composite impact score for a staff member.
        
        Combines frequency and quality metrics into a single score.
        """
        # Normalize publication count (log scale to avoid over-weighting prolific authors)
        pub_score = np.log1p(staff_summary['publication_count']) / np.log1p(100)
        
        # Quality components
        relevance = staff_summary['avg_relevance']
        quality = staff_summary['avg_quality']
        
        # Citations per publication (normalized)
        citations_norm = min(staff_summary['citations_per_pub'] / 50.0, 1.0)
        
        # Composite score with weights
        impact = (
            0.3 * pub_score +          # Publication frequency (log-scaled)
            0.3 * relevance +          # Search relevance
            0.25 * quality +           # Overall quality score
            0.15 * citations_norm      # Citation impact
        )
        
        # Boost for high data completeness
        completeness_bonus = staff_summary['avg_completeness'] * 0.1
        
        return min(impact + completeness_bonus, 1.0)
    
    def get_staff_tier(self, staff_summary: Dict) -> str:
        """Categorize staff into performance tiers.
        
        Args:
            staff_summary: Dictionary with staff metrics
            
        Returns:
            Tier classification string
        """
        pub_count = staff_summary['publication_count']
        quality = staff_summary['avg_quality']
        
        # Define thresholds
        high_pub_threshold = 5
        high_quality_threshold = 0.6
        
        if pub_count >= high_pub_threshold and quality >= high_quality_threshold:
            return "Star Performer"
        elif pub_count < high_pub_threshold and quality >= high_quality_threshold:
            return "Rising Star"
        elif pub_count >= high_pub_threshold and quality < high_quality_threshold:
            return "Prolific Contributor"
        else:
            return "Developing Researcher"