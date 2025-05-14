"""Interactive visualization components for Bibtex Analyzer."""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from ..utils.file_io import ensure_directory_exists

logger = logging.getLogger(__name__)

def create_interactive_plot(
    data: pd.DataFrame,
    tag_column: str = "tags",
    output_file: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """Create an interactive visualization of the data.
    
    This function creates an interactive scatter plot where each point represents a paper,
    with position determined by dimensionality reduction of the tag space.
    
    Args:
        data: DataFrame containing the data
        tag_column: Name of the column containing tags
        output_file: If provided, save the interactive HTML to this file
        **kwargs: Additional keyword arguments passed to plotly.graph_objects.Figure
        
    Returns:
        Plotly Figure object
    """
    try:
        # For now, we'll create a simple visualization based on tag counts
        # In a real implementation, you might want to use dimensionality reduction
        # like UMAP or t-SNE on a tag co-occurrence matrix
        
        # Count tags per paper
        data['tag_count'] = data[tag_column].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )
        
        # Create a simple scatter plot
        fig = px.scatter(
            data,
            x=data.index,
            y='tag_count',
            hover_data=['title', 'year'],
            title='Papers by Tag Count',
            labels={'index': 'Paper', 'tag_count': 'Number of Tags'},
            **kwargs
        )
        
        # Save to file if requested
        if output_file:
            ensure_directory_exists(Path(output_file).parent)
            fig.write_html(output_file, include_plotlyjs='cdn')
            logger.info(f"Interactive plot saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive plot: {str(e)}")
        raise

def create_tag_network(
    data: pd.DataFrame,
    tag_column: str = "tags",
    min_cooccurrence: int = 2,
    output_file: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """Create an interactive network visualization of tag co-occurrence.
    
    Args:
        data: DataFrame containing the data
        tag_column: Name of the column containing tags
        min_cooccurrence: Minimum number of co-occurrences to show an edge
        output_file: If provided, save the interactive HTML to this file
        **kwargs: Additional keyword arguments
        
    Returns:
        Plotly Figure object
    """
    try:
        from itertools import combinations
        import networkx as nx
        
        # Create a co-occurrence matrix
        cooccurrence = {}
        
        for _, row in data.iterrows():
            tags = [tag.strip().title() for tag in str(row[tag_column]).split(",") 
                   if tag.strip()]
            
            # Update co-occurrence counts
            for tag1, tag2 in combinations(sorted(tags), 2):
                if tag1 != tag2:
                    key = tuple(sorted([tag1, tag2]))
                    cooccurrence[key] = cooccurrence.get(key, 0) + 1
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes and edges
        for (tag1, tag2), weight in cooccurrence.items():
            if weight >= min_cooccurrence:
                G.add_edge(tag1, tag2, weight=weight)
        
        # Use spring layout to position nodes
        pos = nx.spring_layout(G)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}")
            node_size.append(len(G.edges(node)) * 2 + 5)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Tag Co-occurrence Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        # Save to file if requested
        if output_file:
            ensure_directory_exists(Path(output_file).parent)
            fig.write_html(output_file, include_plotlyjs='cdn')
            logger.info(f"Tag network visualization saved to {output_file}")
        
        return fig
        
    except ImportError:
        logger.warning("NetworkX not installed. Install with 'pip install networkx' to use network visualizations.")
        return None
    except Exception as e:
        logger.error(f"Error creating tag network: {str(e)}")
        raise
