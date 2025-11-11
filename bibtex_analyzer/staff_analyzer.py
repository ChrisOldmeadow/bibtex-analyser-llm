"""Staff analysis module for ranking researchers based on publication frequency and quality."""

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib
import json
import logging
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from dash import html
import dash_bootstrap_components as dbc
from openai import OpenAI

logger = logging.getLogger(__name__)


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

    def _extract_staff_ids(self, paper: Dict[str, Any]) -> List[str]:
        """Extract normalized staff identifiers from a paper record."""
        staff_ids: List[str] = []
        all_staff = paper.get('all_staff_ids')
        if all_staff:
            if isinstance(all_staff, list):
                staff_ids.extend(all_staff)
            else:
                staff_ids.extend([s.strip() for s in str(all_staff).replace(';', ',').split(',')])
        
        primary_staff = paper.get('staff_id')
        if primary_staff:
            staff_ids.append(str(primary_staff).strip())
        
        cleaned: List[str] = []
        seen = set()
        for sid in staff_ids:
            sid_str = str(sid).strip()
            if not sid_str or sid_str.lower() == 'nan':
                continue
            if sid_str not in seen:
                seen.add(sid_str)
                cleaned.append(sid_str)
        return cleaned
    
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
            'result_ids': set(),
            'total_count': 0,
            'relevance_scores': [],
            'quality_scores': [],
            'completeness_scores': [],
            'total_citations': 0,
            'years': [],
            'first_author_hits': 0,
            'senior_author_hits': 0,
            'intl_collab_hits': 0,
            'clinical_trial_hits': 0
        })
        
        # Process each publication
        for idx, row in search_results.iterrows():
            paper = row.to_dict()
            
            # Get all staff IDs for this publication
            staff_ids = self._extract_staff_ids(paper)
            
            # Skip if no staff identified
            if not staff_ids:
                continue
            
            result_id = paper.get('result_row_id')
            if result_id is None:
                result_id = row.get('result_row_id')
            try:
                result_id_int = int(result_id)
            except (TypeError, ValueError):
                result_id_int = None
            
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
                    if result_id_int is not None:
                        if result_id_int in data['result_ids']:
                            continue
                        data['result_ids'].add(result_id_int)
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

                    if self._is_first_author(paper):
                        data['first_author_hits'] += 1
                    if self._is_senior_author(paper):
                        data['senior_author_hits'] += 1
                    if self._has_international_collab(paper):
                        data['intl_collab_hits'] += 1
                    if self._is_clinical_trial(paper):
                        data['clinical_trial_hits'] += 1
        
        # Create summary DataFrame
        summary_data = []
        for staff_id, data in staff_data.items():
            if data['total_count'] > 0:
                summary = {
                    'staff_id': staff_id,
                    'staff_display_id': self._make_display_id(staff_id),
                    'publication_count': data['total_count'],
                    'avg_relevance': np.mean(data['relevance_scores']) if data['relevance_scores'] else 0,
                    'avg_quality': np.mean(data['quality_scores']) if data['quality_scores'] else 0,
                    'avg_completeness': np.mean(data['completeness_scores']) if data['completeness_scores'] else 0,
                    'total_citations': data['total_citations'],
                    'citations_per_pub': data['total_citations'] / data['total_count'],
                    'year_range': f"{min(data['years'])}-{max(data['years'])}" if data['years'] else "N/A",
                    'active_years': len(set(data['years'])) if data['years'] else 0,
                    'first_author_hits': data['first_author_hits'],
                    'senior_author_hits': data['senior_author_hits'],
                    'intl_collab_hits': data['intl_collab_hits'],
                    'clinical_trial_hits': data['clinical_trial_hits']
                }

                total = max(1, data['total_count'])
                summary['first_author_rate'] = data['first_author_hits'] / total
                summary['senior_author_rate'] = data['senior_author_hits'] / total
                summary['international_collab_rate'] = data['intl_collab_hits'] / total
                summary['clinical_trial_rate'] = data['clinical_trial_hits'] / total
                
                # Calculate composite impact score
                summary['impact_score'] = self._calculate_impact_score(summary)
                
                summary_data.append(summary)
        
        # Create DataFrame and sort by impact score
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values('impact_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
        
        return df

    def map_staff_to_publications(self, search_results: pd.DataFrame) -> Dict[str, List[int]]:
        """Map staff IDs to the result_row_ids of matching publications in the search results."""
        if search_results is None or search_results.empty:
            return {}
        
        mapping: Dict[str, List[int]] = defaultdict(list)
        
        for _, row in search_results.iterrows():
            paper = row.to_dict()
            staff_ids = self._extract_staff_ids(paper)
            if not staff_ids:
                continue
            
            result_id = paper.get('result_row_id')
            if result_id is None:
                result_id = row.get('result_row_id')
            if result_id is None:
                continue
            try:
                result_id = int(result_id)
            except (TypeError, ValueError):
                continue
            
            for staff_id in staff_ids:
                key = str(staff_id).strip()
                if not key or key.lower() == 'nan':
                    continue
                existing = mapping[key]
                if result_id not in existing:
                    existing.append(result_id)
        
        return dict(mapping)
    
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

    def _make_display_id(self, staff_id: str) -> str:
        """Create a short hash-based identifier for UI display."""
        if not staff_id:
            return "STAFF-NA"
        digest = hashlib.sha1(str(staff_id).encode("utf-8")).hexdigest()[:8].upper()
        return f"STAFF-{digest}"

    @staticmethod
    def _truthy(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        value_str = str(value).strip().lower()
        return value_str in {'1', 'y', 'yes', 'true'}

    def _is_first_author(self, paper: Dict[str, Any]) -> bool:
        if self._truthy(paper.get('First_Author')):
            return True
        position = paper.get('Author_Position')
        return str(position).strip() == '1'

    def _is_senior_author(self, paper: Dict[str, Any]) -> bool:
        if self._truthy(paper.get('Last_Author')):
            return True
        try:
            position = int(float(str(paper.get('Author_Position', 0))))
            max_pos = int(float(str(paper.get('Pub_Max_Position', 0))))
            return max_pos > 0 and position == max_pos
        except (ValueError, TypeError):
            return False

    def _has_international_collab(self, paper: Dict[str, Any]) -> bool:
        countries_raw = str(paper.get('Countries', '') or '')
        if not countries_raw.strip():
            return False
        tokens = [token.strip().upper() for token in countries_raw.replace(';', ',').split(',') if token.strip()]
        if not tokens:
            return False
        unique = set(tokens)
        if len(unique) > 1:
            return True
        return any(token not in {'AUSTRALIA'} for token in unique)

    def _is_clinical_trial(self, paper: Dict[str, Any]) -> bool:
        fields = [
            paper.get('Publication_Type', ''),
            paper.get('Keywords', ''),
            paper.get('Output_Title', ''),
            paper.get('Output_abstract', ''),
            paper.get('NURO_abstract', '')
        ]
        haystack = " ".join(str(f or '') for f in fields).lower()
        trial_terms = [
            "clinical trial",
            "randomized",
            "randomised",
            "double-blind",
            "placebo-controlled",
            "phase i",
            "phase ii",
            "phase iii",
            "phase iv",
            "trial registration"
        ]
        return any(term in haystack for term in trial_terms)

    def generate_staff_summary(
        self,
        search_results: pd.DataFrame,
        min_completeness: float = 0.0
    ) -> pd.DataFrame:
        """Wrap aggregate_staff_metrics with defensive checks."""
        if search_results is None or search_results.empty:
            return pd.DataFrame()
        return self.aggregate_staff_metrics(
            search_results=search_results,
            min_completeness=min_completeness
        )

    def render_staff_summary(
        self,
        summary_df: pd.DataFrame,
        max_rows: int = 10
    ) -> html.Div:
        """Render a set of Bootstrap cards summarizing staff performance."""
        if summary_df is None or summary_df.empty:
            return html.Div("No staff contributors found.", className="text-muted")

        cards = []
        for _, row in summary_df.head(max_rows).iterrows():
            tier_badge = dbc.Badge(
                self.get_staff_tier(row),
                color="primary",
                className="ms-2"
            )

            metrics_list = html.Ul([
                html.Li(f"Avg Relevance: {row['avg_relevance']:.2f}"),
                html.Li(f"Avg Quality: {row['avg_quality']:.2f}"),
                html.Li(f"Citations per Publication: {row['citations_per_pub']:.1f}"),
                html.Li(f"Active Years: {row['active_years']} ({row['year_range']})"),
                html.Li(f"Impact Score: {row['impact_score']:.2f}")
            ], className="mb-0")

            summary_blocks: List[html.Div] = []
            if 'llm_summary' in summary_df.columns:
                summary_text = row.get('llm_summary')
                if summary_text and str(summary_text).strip():
                    focus_topics = row.get('llm_focus_topics')
                    if isinstance(focus_topics, str):
                        try:
                            parsed_focus = json.loads(focus_topics)
                        except (json.JSONDecodeError, TypeError):
                            parsed_focus = [focus_topics]
                        focus_topics = parsed_focus
                    focus_list = [
                        html.Span(", ".join(focus_topics), className="text-muted")
                    ] if focus_topics else []
                    summary_blocks = [
                        html.Div([
                            html.Strong("LLM Summary", className="d-block small text-uppercase text-muted"),
                            html.P(str(summary_text), className="mb-1 small"),
                            *focus_list
                        ], className="bg-light rounded p-2 mt-3 staff-llm-summary")
                    ]

            cards.append(
                dbc.Card(
                    [
                        dbc.CardHeader([
                            html.Strong(str(row.get('staff_display_id', row['staff_id']))),
                            tier_badge
                        ]),
                        dbc.CardBody([
                            html.P(
                                f"Publications: {int(row['publication_count'])}",
                                className="mb-2"
                            ),
                            metrics_list,
                            *summary_blocks,
                            dbc.Button(
                                f"View {int(row['publication_count'])} publications",
                                id={'type': 'staff-pub-btn', 'index': str(row['staff_id'])},
                                color="link",
                                size="sm",
                                className="px-0 mt-2"
                            )
                        ])
                    ],
                    className="mb-3"
                )
            )
    
        return html.Div([
            html.Small(
                "Showing top 10 staff by impact score. Download the full CSV for all contributors.",
                className="text-muted d-block mb-3"
            ),
            *cards
        ])


class StaffLLMSummarizer:
    """Generate LLM-based summaries describing how staff contribute to a search topic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        cache_dir: str = ".staff_llm_cache"
    ) -> None:
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def _truthy(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {'1', 'y', 'yes', 'true'}

    def _is_first_author(self, paper: Dict[str, Any]) -> bool:
        if self._truthy(paper.get('First_Author')):
            return True
        position = paper.get('Author_Position')
        return str(position).strip() == '1'

    def _is_senior_author(self, paper: Dict[str, Any]) -> bool:
        if self._truthy(paper.get('Last_Author')):
            return True
        try:
            position = int(float(str(paper.get('Author_Position', 0))))
            max_pos = int(float(str(paper.get('Pub_Max_Position', 0))))
            return max_pos > 0 and position == max_pos
        except (ValueError, TypeError):
            return False

    def _has_international_collab(self, paper: Dict[str, Any]) -> bool:
        countries_raw = str(paper.get('Countries', '') or '')
        if not countries_raw.strip():
            return False
        tokens = [token.strip().upper() for token in countries_raw.replace(';', ',').split(',') if token.strip()]
        if not tokens:
            return False
        unique = set(tokens)
        if len(unique) > 1:
            return True
        return any(token not in {'AUSTRALIA'} for token in unique)

    def _is_clinical_trial(self, paper: Dict[str, Any]) -> bool:
        fields = [
            paper.get('Publication_Type', ''),
            paper.get('Keywords', ''),
            paper.get('Output_Title', ''),
            paper.get('Output_abstract', ''),
            paper.get('NURO_abstract', '')
        ]
        haystack = " ".join(str(f or '') for f in fields).lower()
        trial_terms = [
            "clinical trial",
            "randomized",
            "randomised",
            "double-blind",
            "placebo-controlled",
            "phase i",
            "phase ii",
            "phase iii",
            "phase iv",
            "trial registration"
        ]
        return any(term in haystack for term in trial_terms)

    def _build_cache_key(
        self,
        dataset_id: str,
        query: str,
        staff_id: str,
        paper_ids: List[Any]
    ) -> Path:
        payload = json.dumps(
            {
                "dataset": dataset_id or "uploaded",
                "query": query,
                "staff_id": staff_id,
                "paper_ids": paper_ids
            },
            sort_keys=True
        ).encode("utf-8")
        digest = hashlib.md5(payload).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def summarize_staff(
        self,
        *,
        query: str,
        staff_id: str,
        staff_row: Dict[str, Any],
        papers: pd.DataFrame,
        dataset_id: Optional[str] = None,
        max_papers: int = 5
    ) -> Dict[str, Any]:
        """Return a dict with LLM summary and focus topics."""
        if papers is None or papers.empty:
            return {
                "summary": "No matching publications were available for this staff member.",
                "focus_topics": []
            }

        trimmed = papers.sort_values("search_score", ascending=False).head(max_papers)
        paper_ids = [int(p) for p in trimmed.get("result_row_id", []).tolist() if pd.notna(p)]
        cache_file = self._build_cache_key(dataset_id or "uploaded", query, staff_id, paper_ids)
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass

        staff_label = staff_row.get("staff_display_id") or staff_id
        staff_metrics_parts = [
            f"Impact score {staff_row.get('impact_score', 0):.2f}",
            f"{int(staff_row.get('publication_count', 0))} matched publications"
        ]
        staff_metrics_parts.append(
            f"{int(staff_row.get('first_author_hits', 0))} first-author / "
            f"{int(staff_row.get('senior_author_hits', 0))} senior-author roles"
        )
        staff_metrics_parts.append(
            f"{staff_row.get('international_collab_rate', 0)*100:.0f}% of matches with international collaborators"
        )
        staff_metrics_parts.append(
            f"{staff_row.get('clinical_trial_rate', 0)*100:.0f}% of matches involve clinical trials"
        )
        staff_metrics = "; ".join(staff_metrics_parts)

        paper_lines = []
        for _, paper in trimmed.iterrows():
            title = str(paper.get("title", "Untitled")).strip() or "Untitled paper"
            year = str(paper.get("year", "n.d."))
            journal = str(paper.get("journal", "")).strip()
            abstract = str(paper.get("abstract", "")).replace("\n", " ")
            abstract = abstract[:400] + ("…" if len(abstract) > 400 else "")
            score = float(paper.get("search_score", 0) or 0)
            snippet = f"{title} ({year})"
            if journal:
                snippet += f", {journal}"
            snippet += f" – relevance score {score:.2f}. {abstract}"
            flags = []
            if self._is_first_author(paper):
                flags.append("first author")
            if self._is_senior_author(paper):
                flags.append("senior author")
            if self._has_international_collab(paper):
                flags.append("international collaboration")
            if self._is_clinical_trial(paper):
                flags.append("clinical trial")
            if flags:
                snippet += f" [{' | '.join(flags)}]"
            paper_lines.append(snippet)

        prompt = f"""You are assisting a research evaluation exercise.
Search topic: "{query}"
Staff member: {staff_label} ({staff_metrics})
Matched publications:
- """ + "\n- ".join(paper_lines) + """

Write a concise 2-3 sentence summary explaining how this staff member's matched publications contribute to the search topic. Ground the description in the evidence above: highlight recurring modalities, populations, or techniques (e.g., neuroimaging, digital cognitive testing, wearable sensors) when they are clearly supported; mention international collaborations, leadership roles, or clinical trials only if the papers indicate them. Avoid inventing details that are not implied by the summaries.

Then list three short keywords capturing the dominant focus areas. Respond strictly in JSON with this schema:
{
  "summary": "text",
  "focus_topics": ["keyword1", "keyword2", "keyword3"]
}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=220
            )
            content = response.choices[0].message.content.strip()
            logger.debug("LLM raw response for %s: %s", staff_id, content)
            summary_data = None
            try:
                summary_data = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        summary_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            if summary_data is None:
                logger.warning("LLM did not return parseable JSON for %s. Response: %s", staff_id, content[:200])
                summary_data = {
                    "summary": content.strip(),
                    "focus_topics": []
                }
        except Exception as exc:
            logger.warning("Failed to generate staff summary for %s: %s", staff_id, exc)
            summary_data = {
                "summary": "Unable to generate an LLM summary at this time.",
                "focus_topics": []
            }

        try:
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(summary_data, fh)
        except OSError:
            pass

        return summary_data
