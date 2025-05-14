"""Interactive word cloud visualization using Plotly."""

import logging
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from ..utils.text_processing import slugify
from ..utils.file_io import ensure_directory_exists

logger = logging.getLogger(__name__)

def create_plotly_wordcloud(
    data: pd.DataFrame,
    tag_column: str = "tags",
    width: int = 1000,
    height: int = 800,
    max_words: int = 100,
    output_file: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """Create an interactive word cloud using Plotly.
    
    Args:
        data: DataFrame containing the data
        tag_column: Name of the column containing tags
        width: Width of the output figure
        height: Height of the output figure
        max_words: Maximum number of words to include
        output_file: If provided, save the interactive HTML to this file
        **kwargs: Additional keyword arguments passed to plotly.graph_objects.Figure
        
    Returns:
        Plotly Figure object
    """
    try:
        # Flatten all tags
        all_tags = []
        for tag_list in data[tag_column]:
            tags = [tag.strip().title() for tag in str(tag_list).split(",")]
            all_tags.extend(tags)
        
        # Count tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort and limit to top N tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:max_words]
        tags, counts = zip(*sorted_tags)
        
        # Generate random positions
        x_values = [random.random() for _ in range(len(tags))]
        y_values = [random.random() for _ in range(len(tags))]
        
        # Create hover text
        hover_texts = [
            f"<b>{tag}</b><br>Count: {count}" 
            for tag, count in zip(tags, counts)
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add each word as a separate trace for better interactivity
        for tag, x, y, count, hover_text in zip(tags, x_values, y_values, counts, hover_texts):
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="text",
                text=[tag],
                textfont=dict(
                    size=10 + count * 2,  # Scale size by frequency
                ),
                hovertext=[hover_text],
                hoverinfo="text",
                textposition="middle center",
                hoverlabel=dict(bgcolor="white", font_size=12),
                name=""
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Interactive Word Cloud",
                x=0.5,
                xanchor="center",
                font=dict(size=20)
            ),
            showlegend=False,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=20, r=20, t=60, b=20),
            width=width,
            height=height,
            hovermode="closest",
            **kwargs
        )
        
        # Save to file if requested
        if output_file:
            ensure_directory_exists(Path(output_file).parent)
            fig.write_html(output_file, include_plotlyjs='cdn')
            logger.info(f"Interactive word cloud saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive word cloud: {str(e)}")
        raise

def create_interactive_plot(
    data: pd.DataFrame,
    tag_column: str = "tags",
    output_file: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """Create an interactive plot of tags with links to tag pages.
    
    Args:
        data: DataFrame containing the data
        tag_column: Name of the column containing tags
        output_file: If provided, save the interactive HTML to this file
        **kwargs: Additional keyword arguments
        
    Returns:
        Plotly Figure object
    """
    try:
        # Flatten all tags
        all_tags = []
        for tag_list in data[tag_column]:
            tags = [tag.strip().title() for tag in str(tag_list).split(",")]
            all_tags.extend(tags)
        
        # Count tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get top N tags
        top_n = 50
        top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Generate random positions
        x = [random.random() for _ in top_tags]
        y = [random.random() for _ in top_tags]
        
        # Create hover text with titles
        hover_texts = []
        for tag, count in top_tags.items():
            titles = data[data[tag_column].str.contains(tag, case=False, na=False)]['title'].tolist()
            hover_text = f"<b>{tag}</b><br>Count: {count}<br><br>" + "<br>".join(titles[:5])
            if len(titles) > 5:
                hover_text += "<br>..."
            hover_texts.append(hover_text)
        
        # Create figure
        fig = go.Figure()
        
        # Add each tag as a separate trace for better interactivity
        for (tag, count), x_pos, y_pos, hover_text in zip(top_tags.items(), x, y, hover_texts):
            # Create a URL-friendly slug for the tag
            tag_slug = slugify(tag)
            
            # Create clickable link
            tag_link = f"<a href='/data/tags/{tag_slug}.html'>{tag}</a>"
            
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode="text",
                text=[tag_link],
                textfont=dict(
                    size=10 + count * 2,  # Scale size by frequency
                ),
                hovertext=[hover_text],
                hoverinfo="text",
                textposition="middle center",
                hoverlabel=dict(bgcolor="white", font_size=12),
                name=""
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Interactive Tag Cloud",
                x=0.5,
                xanchor="center",
                font=dict(size=20)
            ),
            showlegend=False,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=20, r=20, t=60, b=20),
            width=1200,
            height=800,
            hovermode="closest",
            **kwargs
        )
        
        # Save to file if requested
        if output_file:
            ensure_directory_exists(Path(output_file).parent)
            fig.write_html(output_file, include_plotlyjs='cdn')
            logger.info(f"Interactive tag cloud saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive tag cloud: {str(e)}")
        raise
