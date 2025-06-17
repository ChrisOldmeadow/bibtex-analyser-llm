"""
Bibtex Analyzer - A tool for analyzing and visualizing BibTeX bibliographies.
"""

from .bibtex_processor import process_bibtex_file
from .tag_generator import TagGenerator
from .semantic_search import SemanticSearcher, HybridSemanticSearcher, search_papers
from .visualization import (
    create_mpl_wordcloud,
    create_plotly_wordcloud,
    plot_tag_frequencies,
    create_tag_network
)

__version__ = "0.1.0"
__all__ = [
    'process_bibtex_file',
    'TagGenerator',
    'SemanticSearcher',
    'HybridSemanticSearcher',
    'search_papers',
    'create_mpl_wordcloud',
    'create_plotly_wordcloud',
    'plot_tag_frequencies',
    'create_tag_network'
]
