"""
Tests for visualization modules.
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from bibtex_analyzer.visualization import (
    create_mpl_wordcloud,
    create_plotly_wordcloud,
    plot_tag_frequencies,
    create_tag_network
)

# Sample data for testing
SAMPLE_DATA = {
    "title": ["Paper 1", "Paper 2", "Paper 3", "Paper 4"],
    "abstract": ["Abstract 1", "Abstract 2", "Abstract 3", "Abstract 4"],
    "tags": [
        "Machine Learning, Deep Learning",
        "Deep Learning, Neural Networks",
        "Machine Learning, Data Science",
        "Data Science, Statistics"
    ],
    "year": ["2020", "2021", "2022", "2023"]
}

SAMPLE_DF = pd.DataFrame(SAMPLE_DATA)

def test_create_mpl_wordcloud():
    """Test creating a Matplotlib word cloud."""
    # Test with default parameters
    wc = create_mpl_wordcloud(SAMPLE_DF, show=False)
    assert wc is not None
    
    # Test with output file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        try:
            output_path = tmp.name
            wc = create_mpl_wordcloud(SAMPLE_DF, output_file=output_path, show=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

def test_plot_tag_frequencies():
    """Test plotting tag frequencies."""
    # Test with default parameters
    plot_tag_frequencies(SAMPLE_DF, show=False)
    
    # Test with output file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        try:
            output_path = tmp.name
            plot_tag_frequencies(SAMPLE_DF, output_file=output_path, show=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

def test_create_plotly_wordcloud():
    """Test creating a Plotly word cloud."""
    # Test with default parameters
    fig = create_plotly_wordcloud(SAMPLE_DF)
    assert fig is not None
    assert len(fig.data) > 0
    
    # Test with output file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        try:
            output_path = tmp.name
            fig = create_plotly_wordcloud(SAMPLE_DF, output_file=output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

def test_create_tag_network():
    """Test creating a tag co-occurrence network."""
    try:
        # Test with default parameters (requires networkx)
        fig = create_tag_network(SAMPLE_DF)
        if fig is not None:  # Only runs if networkx is installed
            assert len(fig.data) > 0
            
            # Test with output file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                try:
                    output_path = tmp.name
                    fig = create_tag_network(SAMPLE_DF, output_file=output_path)
                    assert os.path.exists(output_path)
                    assert os.path.getsize(output_path) > 0
                finally:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
    except ImportError:
        # Skip if networkx is not installed
        pass

def test_visualization_error_handling():
    """Test error handling in visualization functions."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        create_mpl_wordcloud(empty_df)
    
    with pytest.raises(ValueError):
        plot_tag_frequencies(empty_df)
    
    with pytest.raises(ValueError):
        create_plotly_wordcloud(empty_df)
    
    # Test with missing columns
    df_missing = SAMPLE_DF.drop(columns=["tags"])
    with pytest.raises(KeyError):
        create_mpl_wordcloud(df_missing)
