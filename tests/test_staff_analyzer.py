"""Tests for staff analysis functionality."""
import pytest
import pandas as pd
import numpy as np
from bibtex_analyzer.staff_analyzer import StaffAnalyzer


def test_quality_score_with_all_metrics():
    """Test quality score calculation with all metrics available."""
    analyzer = StaffAnalyzer(current_year=2024)
    
    paper = {
        'search_score': 0.8,
        'citations_scopus': '50',
        'year': '2020',
        'clarivate_quartile_rank': 'Q1',
        'altmetrics_score': '25'
    }
    
    score, completeness, metrics = analyzer.calculate_quality_score(paper)
    
    assert 0 <= score <= 1
    assert completeness == 1.0  # All metrics available
    assert len(metrics) == 4
    assert 'relevance' in metrics
    assert 'citations' in metrics
    assert 'journal' in metrics
    assert 'altmetrics' in metrics


def test_quality_score_with_missing_metrics():
    """Test quality score calculation with some missing metrics."""
    analyzer = StaffAnalyzer(current_year=2024)
    
    paper = {
        'search_score': 0.7,
        'year': '2022',
        # Missing citations, journal, and altmetrics
    }
    
    score, completeness, metrics = analyzer.calculate_quality_score(paper)
    
    assert 0 <= score <= 1
    assert completeness == 0.25  # Only 1 of 4 metrics
    assert len(metrics) == 1
    assert 'relevance' in metrics


def test_citation_normalization():
    """Test citation score normalization by publication age."""
    analyzer = StaffAnalyzer(current_year=2024)
    
    # Recent paper with few citations
    recent_paper = {
        'citations_scopus': '5',
        'year': '2023'
    }
    recent_score = analyzer._get_citation_score(recent_paper)
    
    # Old paper with same citations
    old_paper = {
        'citations_scopus': '5',
        'year': '2010'
    }
    old_score = analyzer._get_citation_score(old_paper)
    
    # Recent paper should have higher normalized score
    assert recent_score > old_score


def test_multiple_citation_sources():
    """Test handling of multiple citation sources."""
    analyzer = StaffAnalyzer()
    
    paper = {
        'citations_europe_pubmed': '10',
        'citations_scopus': '25',
        'citations_wos': '20',
        'citations_wos_lite': 'nan',
        'citations_scival': '30'
    }
    
    # Should use maximum value (30)
    score = analyzer._get_citation_score(paper)
    assert score is not None
    assert score > 0


def test_journal_quartile_scoring():
    """Test journal quartile to score conversion."""
    analyzer = StaffAnalyzer()
    
    # Test Clarivate quartile
    paper_q1 = {'clarivate_quartile_rank': 'Q1'}
    assert analyzer._get_journal_score(paper_q1) == 1.0
    
    paper_q2 = {'clarivate_quartile_rank': 'Q2'}
    assert analyzer._get_journal_score(paper_q2) == 0.75
    
    # Test SJR quartile as fallback
    paper_sjr = {'sjr_best_quartile': 'Q3'}
    assert analyzer._get_journal_score(paper_sjr) == 0.5
    
    # Test missing quartile
    paper_none = {}
    assert analyzer._get_journal_score(paper_none) is None


def test_staff_aggregation():
    """Test aggregation of staff metrics from search results."""
    analyzer = StaffAnalyzer(current_year=2024)
    
    # Create sample search results
    data = [
        {
            'title': 'Paper 1',
            'search_score': 0.9,
            'staff_id': 'STAFF001',
            'citations_scopus': '100',
            'year': '2020',
            'clarivate_quartile_rank': 'Q1'
        },
        {
            'title': 'Paper 2',
            'search_score': 0.8,
            'all_staff_ids': 'STAFF001, STAFF002',
            'citations_scopus': '50',
            'year': '2021',
            'clarivate_quartile_rank': 'Q2'
        },
        {
            'title': 'Paper 3',
            'search_score': 0.7,
            'staff_id': 'STAFF002',
            'citations_scopus': '25',
            'year': '2022'
        }
    ]
    
    df = pd.DataFrame(data)
    staff_df = analyzer.aggregate_staff_metrics(df)
    
    assert len(staff_df) == 2  # Two unique staff members
    assert 'STAFF001' in staff_df['staff_id'].values
    assert 'STAFF002' in staff_df['staff_id'].values
    
    # Check STAFF001 metrics (2 publications)
    staff1 = staff_df[staff_df['staff_id'] == 'STAFF001'].iloc[0]
    assert staff1['publication_count'] == 2
    assert staff1['total_citations'] == 150  # 100 + 50
    
    # Check ranking
    assert staff_df.iloc[0]['rank'] == 1
    assert staff_df.iloc[1]['rank'] == 2


def test_impact_score_calculation():
    """Test composite impact score calculation."""
    analyzer = StaffAnalyzer()
    
    staff_summary = {
        'publication_count': 10,
        'avg_relevance': 0.8,
        'avg_quality': 0.7,
        'citations_per_pub': 25.0,
        'avg_completeness': 0.9
    }
    
    impact_score = analyzer._calculate_impact_score(staff_summary)
    
    assert 0 <= impact_score <= 1
    # Should be relatively high given good metrics
    assert impact_score > 0.5


def test_staff_tier_classification():
    """Test classification of staff into performance tiers."""
    analyzer = StaffAnalyzer()
    
    # Star performer
    star = {
        'publication_count': 10,
        'avg_quality': 0.8
    }
    assert analyzer.get_staff_tier(star) == "Star Performer"
    
    # Rising star
    rising = {
        'publication_count': 3,
        'avg_quality': 0.7
    }
    assert analyzer.get_staff_tier(rising) == "Rising Star"
    
    # Prolific contributor
    prolific = {
        'publication_count': 8,
        'avg_quality': 0.4
    }
    assert analyzer.get_staff_tier(prolific) == "Prolific Contributor"
    
    # Developing researcher
    developing = {
        'publication_count': 2,
        'avg_quality': 0.3
    }
    assert analyzer.get_staff_tier(developing) == "Developing Researcher"


def test_empty_results_handling():
    """Test handling of empty search results."""
    analyzer = StaffAnalyzer()
    
    empty_df = pd.DataFrame()
    staff_df = analyzer.aggregate_staff_metrics(empty_df)
    
    assert staff_df.empty
    assert len(staff_df) == 0


def test_missing_staff_id_handling():
    """Test handling of publications without staff IDs."""
    analyzer = StaffAnalyzer()
    
    data = [
        {
            'title': 'Paper 1',
            'search_score': 0.9,
            # No staff_id or all_staff_ids
        },
        {
            'title': 'Paper 2',
            'search_score': 0.8,
            'staff_id': 'STAFF001'
        }
    ]
    
    df = pd.DataFrame(data)
    staff_df = analyzer.aggregate_staff_metrics(df)
    
    # Should only include papers with staff IDs
    assert len(staff_df) == 1
    assert staff_df.iloc[0]['staff_id'] == 'STAFF001'