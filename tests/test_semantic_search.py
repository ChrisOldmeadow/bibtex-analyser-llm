"""Tests for semantic search functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from bibtex_analyzer.semantic_search import SemanticSearcher, search_papers


class TestSemanticSearcher:
    """Test the SemanticSearcher class."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        searcher = SemanticSearcher(api_key="test_key")
        assert searcher.api_key == "test_key"
        assert searcher.model == "text-embedding-3-small"
    
    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                SemanticSearcher()
    
    def test_prepare_paper_text(self):
        """Test paper text preparation."""
        searcher = SemanticSearcher(api_key="test_key")
        
        entry = {
            'title': 'Test Paper',
            'abstract': 'This is a test abstract.',
            'tags': 'machine learning, AI',
            'keywords': 'test, paper'
        }
        
        text = searcher.prepare_paper_text(entry)
        expected = 'Test Paper This is a test abstract. machine learning, AI test, paper'
        assert text == expected
    
    def test_prepare_paper_text_empty_fields(self):
        """Test paper text preparation with empty fields."""
        searcher = SemanticSearcher(api_key="test_key")
        
        entry = {
            'title': 'Test Paper',
            'abstract': '',
            'tags': None
        }
        
        text = searcher.prepare_paper_text(entry)
        assert text == 'Test Paper'
    
    def test_exact_search(self, sample_dataframe):
        """Test exact search functionality."""
        searcher = SemanticSearcher(api_key="test_key")
        
        results = searcher.exact_search("machine learning", sample_dataframe)
        
        # Should find the entry with "Machine Learning" in title
        assert len(results) == 1
        assert results[0][1] == 1.0  # Perfect match score
    
    def test_exact_search_case_insensitive(self, sample_dataframe):
        """Test that exact search is case insensitive."""
        searcher = SemanticSearcher(api_key="test_key")
        
        results = searcher.exact_search("MACHINE LEARNING", sample_dataframe)
        
        assert len(results) == 1
        assert results[0][1] == 1.0
    
    def test_fuzzy_search(self, sample_dataframe):
        """Test fuzzy search functionality."""
        searcher = SemanticSearcher(api_key="test_key")
        
        # Search with a typo
        results = searcher.fuzzy_search("machne learning", sample_dataframe, threshold=70.0)
        
        # Should still find the machine learning paper
        assert len(results) >= 1
        # Score should be less than 1 due to typo
        assert all(score < 1.0 for _, score in results)
    
    @patch('bibtex_analyzer.semantic_search.SemanticSearcher.get_embedding')
    @patch('bibtex_analyzer.semantic_search.SemanticSearcher.get_embeddings_batch')
    def test_semantic_search(self, mock_batch_embeddings, mock_single_embedding, sample_dataframe):
        """Test semantic search functionality."""
        searcher = SemanticSearcher(api_key="test_key")
        
        # Mock embeddings
        query_embedding = np.array([1.0, 0.0, 0.0])
        paper_embeddings = np.array([
            [0.9, 0.1, 0.0],  # High similarity
            [0.1, 0.9, 0.0],  # Low similarity
            [0.8, 0.2, 0.0]   # Medium similarity
        ])
        
        mock_single_embedding.return_value = query_embedding
        mock_batch_embeddings.return_value = paper_embeddings
        
        results = searcher.semantic_search("test query", sample_dataframe, threshold=0.7)
        
        # Should find papers with similarity >= 0.7
        assert len(results) == 2  # First and third papers
        assert all(score >= 0.7 for _, score in results)
    
    @patch('bibtex_analyzer.semantic_search.SemanticSearcher.exact_search')
    @patch('bibtex_analyzer.semantic_search.SemanticSearcher.fuzzy_search')
    @patch('bibtex_analyzer.semantic_search.SemanticSearcher.semantic_search')
    def test_multi_search(self, mock_semantic, mock_fuzzy, mock_exact, sample_dataframe):
        """Test multi-search functionality."""
        searcher = SemanticSearcher(api_key="test_key")
        
        # Mock individual search results
        mock_exact.return_value = [(0, 1.0)]
        mock_fuzzy.return_value = [(0, 0.9), (1, 0.8)]
        mock_semantic.return_value = [(0, 0.85), (2, 0.75)]
        
        results = searcher.multi_search(
            "test query",
            sample_dataframe,
            methods=["exact", "fuzzy", "semantic"]
        )
        
        # Should combine results from all methods
        assert not results.empty
        assert 'search_score' in results.columns
        assert 'exact_score' in results.columns
        assert 'fuzzy_score' in results.columns
        assert 'semantic_score' in results.columns
    
    def test_cache_functionality(self):
        """Test embedding caching."""
        searcher = SemanticSearcher(api_key="test_key")
        
        # Test cache key generation
        cache_key = searcher._get_cache_key("test text")
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
        
        # Test same text produces same cache key
        cache_key2 = searcher._get_cache_key("test text")
        assert cache_key == cache_key2


class TestSearchPapers:
    """Test the high-level search_papers function."""
    
    @patch('bibtex_analyzer.semantic_search.SemanticSearcher')
    def test_search_papers_csv(self, mock_searcher_class):
        """Test search_papers with CSV input."""
        # Create a temporary CSV file
        import tempfile
        import os
        
        sample_data = pd.DataFrame([
            {'title': 'Test Paper', 'abstract': 'Test abstract', 'author': 'Test Author'}
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            
            try:
                # Mock the searcher
                mock_searcher = Mock()
                mock_searcher.multi_search.return_value = pd.DataFrame([
                    {'title': 'Test Paper', 'search_score': 0.9}
                ])
                mock_searcher_class.return_value = mock_searcher
                
                # Test the function
                results = search_papers(
                    query="test",
                    input_file=f.name,
                    methods=["exact"]
                )
                
                assert not results.empty
                assert 'search_score' in results.columns
                
            finally:
                os.unlink(f.name)
    
    def test_search_papers_invalid_file(self):
        """Test search_papers with invalid file format."""
        with pytest.raises(ValueError, match="Input file must be CSV or Excel format"):
            search_papers("test", "invalid.txt")


# Fixtures for testing
@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame([
        {
            'title': 'Machine Learning for Beginners',
            'abstract': 'An introduction to machine learning concepts',
            'author': 'John Doe',
            'year': 2020,
            'tags': 'machine learning, tutorial'
        },
        {
            'title': 'Deep Learning Advances',
            'abstract': 'Recent advances in deep learning',
            'author': 'Jane Smith',
            'year': 2021,
            'tags': 'deep learning, neural networks'
        },
        {
            'title': 'Data Science Applications',
            'abstract': 'Practical applications of data science',
            'author': 'Bob Johnson',
            'year': 2022,
            'tags': 'data science, applications'
        }
    ])