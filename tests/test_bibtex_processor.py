"""
Tests for the BibtexProcessor class.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from bibtex_analyzer.bibtex_processor import BibtexProcessor

# Sample BibTeX content for testing
SAMPLE_BIBTEX = """@article{key1,
  title = {Sample Title 1},
  author = {Author, A. and Author, B.},
  journal = {Journal of Testing},
  year = {2023},
  abstract = {This is a test abstract.}
}

@inproceedings{key2,
  title = {Sample Title 2},
  author = {Author, C. and Author, D.},
  booktitle = {Proceedings of the Test Conference},
  year = {2022},
  abstract = {Another test abstract.}
}
"""


def test_load_entries():
    """Test loading entries from a BibTeX file."""
    with patch('builtins.open', mock_open(read_data=SAMPLE_BIBTEX)) as mock_file:
        processor = BibtexProcessor()
        entries = processor.load_entries("dummy.bib")
        
        assert len(entries) == 2
        assert entries[0]["title"] == "Sample Title 1"
        assert entries[1]["title"] == "Sample Title 2"
        assert "abstract" in entries[0]
        assert "abstract" in entries[1]


def test_get_entry_by_id():
    """Test retrieving an entry by its ID."""
    processor = BibtexProcessor()
    processor.entries = [
        {"id": "key1", "title": "Title 1"},
        {"id": "key2", "title": "Title 2"}
    ]
    
    entry = processor.get_entry_by_id("key2")
    assert entry["title"] == "Title 2"
    
    entry = processor.get_entry_by_id("nonexistent")
    assert entry is None


def test_filter_entries():
    """Test filtering entries by year and required fields."""
    processor = BibtexProcessor()
    processor.entries = [
        {"id": "key1", "title": "Title 1", "year": "2020"},
        {"id": "key2", "title": "Title 2", "year": "2021"},
        {"id": "key3", "title": "Title 3", "year": "2022"},
        {"id": "key4", "title": "Title 4", "year": "2021", "journal": "Test Journal"}
    ]
    
    # Test year filtering
    filtered = processor.filter_entries(min_year=2021)
    assert len(filtered) == 3
    assert all(int(entry["year"]) >= 2021 for entry in filtered)
    
    # Test required fields
    filtered = processor.filter_entries(required_fields=["journal"])
    assert len(filtered) == 1
    assert filtered[0]["id"] == "key4"
    
    # Test combined filters
    filtered = processor.filter_entries(min_year=2021, max_year=2021, required_fields=["journal"])
    assert len(filtered) == 1
    assert filtered[0]["id"] == "key4"


def test_export_to_csv():
    """Test exporting entries to a CSV file."""
    import pandas as pd
    import tempfile
    
    processor = BibtexProcessor()
    processor.entries = [
        {"id": "key1", "title": "Title 1", "year": "2020"},
        {"id": "key2", "title": "Title 2", "year": "2021"}
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        try:
            output_path = tmp.name
            
            # Test export
            result_path = processor.export_to_csv(output_path)
            assert result_path == output_path
            
            # Verify file contents
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert list(df.columns) == ["id", "title", "year"]
            assert df.iloc[0]["title"] == "Title 1"
            
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)


def test_get_field_values():
    """Test getting values for a specific field."""
    processor = BibtexProcessor()
    processor.entries = [
        {"id": "key1", "title": "Title 1", "year": "2020"},
        {"id": "key2", "title": "Title 2", "year": "2021"},
        {"id": "key3", "title": "Title 1", "year": "2022"}
    ]
    
    # Test unique values
    titles = processor.get_field_values("title")
    assert set(titles) == {"Title 1", "Title 2"}
    
    # Test non-unique values
    years = processor.get_field_values("year", unique=False)
    assert len(years) == 3
    assert "2020" in years
    
    # Test non-existent field
    empty = processor.get_field_values("nonexistent")
    assert empty == []


def test_process_bibtex_file():
    """Test the convenience function for processing a BibTeX file."""
    with patch('builtins.open', mock_open(read_data=SAMPLE_BIBTEX)) as mock_file:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                output_path = tmp.name
                
                # Test processing and saving
                entries = process_bibtex_file("dummy.bib", output_path)
                
                assert len(entries) == 2
                assert os.path.exists(output_path)
                
                # Verify CSV contents
                import pandas as pd
                df = pd.read_csv(output_path)
                assert len(df) == 2
                assert "title" in df.columns
                assert "abstract" in df.columns
                
            finally:
                # Clean up
                if os.path.exists(output_path):
                    os.unlink(output_path)
