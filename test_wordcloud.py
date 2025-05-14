"""Test script for the interactive word cloud functionality."""
import pandas as pd
from pathlib import Path
from bibtex_analyzer.visualization.wordcloud2 import create_interactive_wordcloud

def create_test_data() -> pd.DataFrame:
    """Create a test DataFrame with sample data."""
    data = [
        {
            'title': 'Paper 1',
            'author': 'Author A and Author B',
            'year': '2022',
            'url': 'https://example.com/paper1',
            'tags': 'machine learning, deep learning, nlp'
        },
        {
            'title': 'Paper 2',
            'author': 'Author C and Author D',
            'year': '2021',
            'url': 'https://example.com/paper2',
            'tags': 'deep learning, computer vision, cnn'
        },
        {
            'title': 'Paper 3',
            'author': 'Author E and Author F',
            'year': '2023',
            'url': 'https://example.com/paper3',
            'tags': 'nlp, transformers, attention'
        },
        {
            'title': 'Paper 4',
            'author': 'Author G and Author H',
            'year': '2022',
            'url': 'https://example.com/paper4',
            'tags': 'reinforcement learning, rl, deep learning'
        },
        {
            'title': 'Paper 5',
            'author': 'Author I and Author J',
            'year': '2021',
            'url': 'https://example.com/paper5',
            'tags': 'computer vision, object detection, yolo'
        },
        {
            'title': 'Paper 6',
            'author': 'Author K and Author L',
            'year': '2023',
            'url': 'https://example.com/paper6',
            'tags': 'nlp, llm, gpt, transformers'
        },
        {
            'title': 'Paper 7',
            'author': 'Author M and Author N',
            'year': '2022',
            'url': 'https://example.com/paper7',
            'tags': 'reinforcement learning, robotics, deep learning'
        },
        {
            'title': 'Paper 8',
            'author': 'Author O and Author P',
            'year': '2021',
            'url': 'https://example.com/paper8',
            'tags': 'computer vision, segmentation, unet'
        },
        {
            'title': 'Paper 9',
            'author': 'Author Q and Author R',
            'year': '2023',
            'url': 'https://example.com/paper9',
            'tags': 'nlp, sentiment analysis, bert'
        },
        {
            'title': 'Paper 10',
            'author': 'Author S and Author T',
            'year': '2022',
            'url': 'https://example.com/paper10',
            'tags': 'machine learning, time series, forecasting'
        },
    ]
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create test data
    df = create_test_data()
    
    # Generate interactive word cloud
    output_file = "test_wordcloud.html"
    create_interactive_wordcloud(
        df,
        output_file=output_file,
        tag_column='tags',
        title='Test Interactive Word Cloud',
        width=1000,
        height=800
    )
    
    print(f"Interactive word cloud saved to {output_file}")
