"""
Shared test fixtures and configuration.
"""

import pytest
import pandas as pd
import numpy as np

# Sample data for testing
@pytest.fixture
def sample_entries():
    """Sample BibTeX entries for testing."""
    return [
        {
            "id": "key1",
            "title": "Machine Learning for Beginners",
            "author": "Author A, Author B",
            "year": "2020",
            "journal": "Journal of ML",
            "abstract": "An introduction to machine learning concepts and applications.",
            "tags": "Machine Learning, Tutorial, Beginners"
        },
        {
            "id": "key2",
            "title": "Deep Learning Advances",
            "author": "Author C, Author D",
            "year": "2021",
            "journal": "Neural Networks",
            "abstract": "Recent advances in deep learning architectures.",
            "tags": "Deep Learning, Neural Networks, AI"
        },
        {
            "id": "key3",
            "title": "Data Science in Practice",
            "author": "Author E, Author F",
            "year": "2022",
            "journal": "Data Science Journal",
            "abstract": "Practical applications of data science in industry.",
            "tags": "Data Science, Applications, Industry"
        }
    ]

@pytest.fixture
def sample_dataframe(sample_entries):
    """Sample DataFrame for testing."""
    return pd.DataFrame(sample_entries)

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    class MockChoice:
        def __init__(self, text):
            self.text = text
            self.finish_reason = "stop"
            self.index = 0
    
    class MockResponse:
        def __init__(self, text):
            self.choices = [MockChoice(text)]
    
    return MockResponse("Mocked response")

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test output."""
    return tmp_path

@pytest.fixture
def mock_bibtex_file(tmp_path, sample_entries):
    """Create a temporary BibTeX file for testing."""
    bib_content = """@article{key1,
  title = {Machine Learning for Beginners},
  author = {Author A and Author B},
  journal = {Journal of ML},
  year = {2020},
  abstract = {An introduction to machine learning concepts and applications.}
}

@article{key2,
  title = {Deep Learning Advances},
  author = {Author C and Author D},
  journal = {Neural Networks},
  year = {2021},
  abstract = {Recent advances in deep learning architectures.}
}
"""
    file_path = tmp_path / "test.bib"
    file_path.write_text(bib_content)
    return str(file_path)
