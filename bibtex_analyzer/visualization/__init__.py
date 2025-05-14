"""
Visualization module for Bibtex Analyzer.
"""
import os
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

from .wordcloud2 import create_interactive_wordcloud

def create_mpl_wordcloud(tags: List[str], output_path: Optional[str] = None) -> None:
    """Create a word cloud using matplotlib."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tags))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def create_plotly_wordcloud(df: pd.DataFrame, text_col: str = 'tags', 
                          title: str = 'Word Cloud of Tags') -> go.Figure:
    """Create an interactive word cloud using plotly."""
    # Flatten tags and count frequencies
    all_tags = []
    for tags in df[text_col].dropna():
        if isinstance(tags, str):
            all_tags.extend([tag.strip() for tag in tags.split(',')])
    
    tag_counts = Counter(all_tags)
    
    # Create figure
    fig = go.Figure()
    
    # Add words as scatter plot
    words = list(tag_counts.keys())
    counts = [tag_counts[word] for word in words]
    
    fig.add_trace(go.Scatter(
        x=[0] * len(words),
        y=[0] * len(words),
        text=words,
        mode='text',
        textfont=dict(
            size=[min(10 + count * 3, 50) for count in counts],
            color=[f'rgb({i%200}, {i%100+100}, {i%150+50})' for i in range(len(words))]
        ),
        hovertext=[f'{word}: {count} occurrences' for word, count in zip(words, counts)],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def plot_tag_frequencies(df: pd.DataFrame, tag_column: str = 'tags', 
                        top_n: int = 20) -> go.Figure:
    """Plot the frequency of tags as a bar chart."""
    # Flatten tags and count frequencies
    all_tags = []
    for tags in df[tag_column].dropna():
        if isinstance(tags, str):
            all_tags.extend([tag.strip() for tag in tags.split(',')])
    
    tag_counts = Counter(all_tags)
    common_tags = tag_counts.most_common(top_n)
    
    # Create bar chart
    fig = px.bar(
        x=[tag for tag, _ in common_tags],
        y=[count for _, count in common_tags],
        labels={'x': 'Tag', 'y': 'Frequency'},
        title=f'Top {top_n} Most Common Tags'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title='Tag',
        yaxis_title='Frequency',
        height=600
    )
    
    return fig

def create_tag_network(df: pd.DataFrame, tag_column: str = 'tags', 
                      min_cooccurrence: int = 2) -> Optional[go.Figure]:
    """Create a network graph of co-occurring tags."""
    # Extract and count tag co-occurrences
    tag_sets = []
    for tags in df[tag_column].dropna():
        if isinstance(tags, str):
            tag_list = [tag.strip() for tag in tags.split(',')]
            if len(tag_list) > 1:  # Only include entries with multiple tags
                tag_sets.append(set(tag_list))
    
    if not tag_sets:
        return None
    
    # Count co-occurrences
    cooccurrence = Counter()
    for tag_set in tag_sets:
        tags = sorted(tag_set)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                cooccurrence[(tags[i], tags[j])] += 1
    
    # Filter by minimum co-occurrence
    edges = [(src, dst, count) for (src, dst), count in cooccurrence.items() 
             if count >= min_cooccurrence]
    
    if not edges:
        return None
    
    # Create network
    G = nx.Graph()
    for src, dst, count in edges:
        G.add_edge(src, dst, weight=count)
    
    # Get node positions
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
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'<b>{node}</b>')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    # Color node points by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(G.adj[node]))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='<br>Tag Network Graph',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

__all__ = [
    'create_mpl_wordcloud',
    'create_plotly_wordcloud',
    'create_interactive_wordcloud',
    'plot_tag_frequencies',
    'create_tag_network'
]
