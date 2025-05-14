"""Word cloud visualization using Matplotlib."""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

from ..utils.text_processing import collapse_tags, filter_redundant_tags

logger = logging.getLogger(__name__)

def create_mpl_wordcloud(
    data: pd.DataFrame,
    tag_column: str = "tags",
    width: int = 1200,
    height: int = 800,
    background_color: str = "white",
    max_words: int = 100,
    colormap: str = "viridis",
    output_file: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Optional[WordCloud]:
    """Create a word cloud visualization using Matplotlib.
    
    Args:
        data: DataFrame containing the data
        tag_column: Name of the column containing tags
        width: Width of the output image
        height: Height of the output image
        background_color: Background color of the word cloud
        max_words: Maximum number of words to include
        colormap: Matplotlib colormap to use
        output_file: If provided, save the word cloud to this file
        show: Whether to display the plot
        **kwargs: Additional keyword arguments passed to WordCloud
        
    Returns:
        WordCloud object if show=False, None otherwise
    """
    try:
        # Flatten all tags
        all_tags = []
        for tag_list in data[tag_column]:
            tags = [tag.strip().title() for tag in str(tag_list).split(",")]
            all_tags.extend(tags)
        
        # Collapse similar tags
        collapsed_tags = collapse_tags(all_tags)
        
        # Count tag frequencies
        tag_counts = {}
        for tag in collapsed_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Filter redundant tags
        filtered_counts = filter_redundant_tags(
            tag_counts,
            drop_if_subset_of={"Randomised Controlled Trial", "Cohort Study"}
        )
        
        # Limit to top N tags
        sorted_tags = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
        top_tags = dict(sorted_tags[:max_words])
        
        # Generate word cloud
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            **kwargs
        )
        
        wc.generate_from_frequencies(top_tags)
        
        # Plot
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300, quality=100)
            logger.info(f"Word cloud saved to {output_file}")
        
        if show:
            plt.show()
            return None
        
        return wc
        
    except Exception as e:
        logger.error(f"Error creating word cloud: {str(e)}")
        raise

def plot_tag_frequencies(
    data: pd.DataFrame,
    tag_column: str = "tags",
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    output_file: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot a bar chart of the most frequent tags.
    
    Args:
        data: DataFrame containing the data
        tag_column: Name of the column containing tags
        top_n: Number of top tags to show
        figsize: Figure size (width, height)
        output_file: If provided, save the plot to this file
        show: Whether to display the plot
    """
    try:
        # Flatten and count tags
        all_tags = []
        for tag_list in data[tag_column]:
            tags = [tag.strip().title() for tag in str(tag_list).split(",")]
            all_tags.extend(tags)
        
        # Count and sort tags
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get top N tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        tags, counts = zip(*sorted_tags)
        
        # Create horizontal bar chart
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(tags))
        
        plt.barh(y_pos, counts, align='center', alpha=0.7)
        plt.yticks(y_pos, tags)
        plt.xlabel('Frequency')
        plt.title(f'Top {top_n} Most Frequent Tags')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Tag frequencies plot saved to {output_file}")
        
        if show:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting tag frequencies: {str(e)}")
        raise
