"""
Tests for the command-line interface.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import pandas as pd

from bibtex_analyzer.__main__ import parse_args, analyze_command, visualize_command

# Sample data for testing
SAMPLE_ENTRIES = [
    {"id": "key1", "title": "Title 1", "abstract": "Abstract 1", "tags": "Tag1, Tag2"},
    {"id": "key2", "title": "Title 2", "abstract": "Abstract 2", "tags": "Tag2, Tag3"}
]


def test_parse_args():
    """Test command-line argument parsing."""
    # Test analyze command
    args = parse_args(["analyze", "input.bib", "--output", "output.csv"])
    assert args.command == "analyze"
    assert args.input == "input.bib"
    assert args.output == "output.csv"
    
    # Test visualize command
    args = parse_args(["visualize", "input.csv", "--interactive"])
    assert args.command == "visualize"
    assert args.input == "input.csv"
    assert args.interactive is True
    
    # Test dashboard command
    args = parse_args(["dashboard", "--port", "8080"])
    assert args.command == "dashboard"
    assert args.port == 8080


@patch('bibtex_analyzer.__main__.process_bibtex_file')
@patch('bibtex_analyzer.__main__.TagGenerator')
def test_analyze_command(mock_tagger, mock_process):
    """Test the analyze command."""
    # Setup mocks
    mock_process.return_value = SAMPLE_ENTRIES
    mock_tagger.return_value.interactive_tag_generation.return_value = ["Tag1", "Tag2"]
    mock_tagger.return_value.assign_tags_to_abstracts.return_value = SAMPLE_ENTRIES
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        try:
            output_path = tmp.name
            
            # Create test args
            class Args:
                input = "input.bib"
                output = output_path
                samples = 10
                model = "gpt-3.5-turbo"
                methods_model = "gpt-4"
            
            # Run the command
            analyze_command(Args())
            
            # Verify the output file was created
            assert os.path.exists(output_path)
            
            # Verify the mocks were called
            mock_process.assert_called_once_with("input.bib", None)
            mock_tagger.return_value.interactive_tag_generation.assert_called_once()
            mock_tagger.return_value.assign_tags_to_abstracts.assert_called_once()
            
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)


@patch('bibtex_analyzer.__main__.pd')
@patch('bibtex_analyzer.__main__.create_plotly_wordcloud')
@patch('bibtex_analyzer.__main__.create_mpl_wordcloud')
@patch('bibtex_analyzer.__main__.plot_tag_frequencies')
@patch('bibtex_analyzer.__main__.create_tag_network')
def test_visualize_command(
    mock_network, mock_freq, mock_mpl, mock_plotly, mock_pd,
    tmp_path
):
    """Test the visualize command."""
    # Setup mocks
    mock_pd.read_csv.return_value = pd.DataFrame(SAMPLE_ENTRIES)
    mock_plotly.return_value = MagicMock()
    mock_mpl.return_value = MagicMock()
    mock_freq.return_value = None
    mock_network.return_value = MagicMock()
    
    # Create test output directory
    output_dir = tmp_path / "output"
    
    # Test with interactive flag
    class InteractiveArgs:
        input = "input.csv"
        output_dir = str(output_dir)
        interactive = True
        static = False
        network = False
    
    visualize_command(InteractiveArgs())
    
    # Verify interactive visualizations were created
    mock_plotly.assert_called_once()
    assert (output_dir / "wordcloud_interactive.html").exists()
    
    # Test with static flag
    mock_plotly.reset_mock()
    
    class StaticArgs:
        input = "input.csv"
        output_dir = str(output_dir)
        interactive = False
        static = True
        network = False
    
    visualize_command(StaticArgs())
    
    # Verify static visualizations were created
    mock_mpl.assert_called_once()
    mock_freq.assert_called_once()
    assert (output_dir / "wordcloud.png").exists()
    assert (output_dir / "tag_frequencies.png").exists()
    
    # Test with network flag
    if mock_network.return_value is not None:
        mock_network.reset_mock()
        
        class NetworkArgs:
            input = "input.csv"
            output_dir = str(output_dir)
            interactive = False
            static = False
            network = True
        
        visualize_command(NetworkArgs())
        
        # Verify network visualization was created
        mock_network.assert_called_once()
        assert (output_dir / "tag_network.html").exists()


@patch('bibtex_analyzer.__main__.dash')
@patch('bibtex_analyzer.__main__.pd')
def test_dashboard_command(mock_pd, mock_dash, tmp_path):
    """Test the dashboard command."""
    # Setup mocks
    mock_app = MagicMock()
    mock_dash.Dash.return_value = mock_app
    mock_pd.read_csv.return_value = pd.DataFrame(SAMPLE_ENTRIES)
    
    # Test with default port
    class Args:
        port = 8050
        debug = False
    
    dashboard_command(Args())
    
    # Verify the app was created and run
    mock_dash.Dash.assert_called_once()
    mock_app.run_server.assert_called_once_with(debug=False, port=8050)


def test_main_no_args(capsys):
    """Test the main function with no arguments."""
    with patch('sys.argv', ['bibtex_analyzer']):
        with pytest.raises(SystemExit):
            from bibtex_analyzer.__main__ import main
            main()
    
    # Verify help message was printed
    captured = capsys.readouterr()
    assert "usage:" in captured.out


@patch('bibtex_analyzer.__main__.analyze_command')
def test_main_analyze(mock_analyze):
    """Test the main function with analyze command."""
    with patch('sys.argv', ['bibtex_analyzer', 'analyze', 'input.bib']):
        from bibtex_analyzer.__main__ import main
        main()
    
    mock_analyze.assert_called_once()


@patch('bibtex_analyzer.__main__.visualize_command')
def test_main_visualize(mock_visualize):
    """Test the main function with visualize command."""
    with patch('sys.argv', ['bibtex_analyzer', 'visualize', 'input.csv']):
        from bibtex_analyzer.__main__ import main
        main()
    
    mock_visualize.assert_called_once()


@patch('bibtex_analyzer.__main__.dashboard_command')
def test_main_dashboard(mock_dashboard):
    """Test the main function with dashboard command."""
    with patch('sys.argv', ['bibtex_analyzer', 'dashboard']):
        from bibtex_analyzer.__main__ import main
        main()
    
    mock_dashboard.assert_called_once()
