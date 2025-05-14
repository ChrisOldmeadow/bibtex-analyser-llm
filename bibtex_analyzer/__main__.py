"""Bibtex Analyzer - Command Line Interface.

This module provides a command-line interface for the Bibtex Analyzer tool.
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from . import __version__
from .bibtex_processor import BibtexProcessor, process_bibtex_file
from .tag_generator import TagGenerator
from .visualization import (
    create_mpl_wordcloud,
    create_plotly_wordcloud,
    create_tag_network,
    plot_tag_frequencies
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bibtex_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Bibtex Analyzer - Analyze and visualize BibTeX bibliographies')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                             help='Model to use for tag generation (default: gpt-3.5-turbo)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a BibTeX file and generate tags', parents=[parent_parser])
    analyze_parser.add_argument('input', type=str, help='Input BibTeX file')
    analyze_parser.add_argument('--output', '-o', type=str, default='tagged_abstracts.csv',
                              help='Output CSV file (default: tagged_abstracts.csv)')
    analyze_parser.add_argument('--tag-samples', type=int, default=30,
                              help='Number of samples to use for tag generation (default: 30)')
    analyze_parser.add_argument('--subset-size', type=int, default=100,
                              help='Process only a subset of entries (0 for all, default: 100)')
    analyze_parser.add_argument('--wordcloud', type=str, nargs='?', const='png', choices=['png', 'html', 'both'],
                              help='Generate word cloud (default: png if no value provided, omit to skip)')
    analyze_parser.add_argument(
        "--methods-model",
        default="gpt-4",
        help="Model to use for statistical methods (default: gpt-4)"
    )
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate visualizations from tagged data"
    )
    visualize_parser.add_argument(
        "input",
        help="Input CSV file with tagged abstracts"
    )
    visualize_parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory for visualizations (default: output)"
    )
    visualize_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive visualizations (Plotly)"
    )
    visualize_parser.add_argument(
        "--static",
        action="store_true",
        help="Generate static visualizations (Matplotlib)"
    )
    visualize_parser.add_argument(
        "--network",
        action="store_true",
        help="Generate tag co-occurrence network"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch interactive dashboard"
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)"
    )
    dashboard_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    # If no arguments provided, show help
    if len(args) == 0:
        parser.print_help()
        sys.exit(0)
    
    return parser.parse_args(args)

def analyze_command(args: argparse.Namespace) -> None:
    """Handle the analyze command.
    
    Args:
        args: Parsed command line arguments
    """
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the BibTeX file
    logger.info(f"Processing BibTeX file: {args.input}")
    entries = process_bibtex_file(args.input)
    
    # Filter out entries without abstracts
    entries_with_abstracts = [e for e in entries if e.get('abstract')]
    if len(entries_with_abstracts) < len(entries):
        logger.warning(f"Skipped {len(entries) - len(entries_with_abstracts)} entries without abstracts")
    
    if not entries_with_abstracts:
        logger.error("No entries with abstracts found. Cannot proceed with tag generation.")
        return
    
    # Initialize tag generator
    tagger = TagGenerator(model=args.model)
    
    # If subset_size is specified, only process that many entries
    if args.subset_size > 0:
        import random
        # Limit the number of entries to process
        num_to_process = min(args.subset_size, len(entries_with_abstracts))
        logger.info(f"Processing a random subset of {num_to_process} entries (from {len(entries_with_abstracts)} with abstracts)")
        entries_to_tag = random.sample(entries_with_abstracts, num_to_process)
    else:
        # Process all entries with abstracts
        entries_to_tag = entries_with_abstracts
    
    # Calculate sample size for tag generation (minimum of tag_samples and available entries)
    tag_sample_size = min(args.tag_samples, len(entries_to_tag))
    
    # Generate tags using the selected entries
    logger.info(f"Generating tags using {tag_sample_size} samples from {len(entries_to_tag)} entries...")
    tags = tagger.interactive_tag_generation(
        entries_to_tag,
        start_n=tag_sample_size,
        max_n=min(tag_sample_size * 2, len(entries_to_tag)),
        step=max(1, tag_sample_size // 3)
    )
    
    if not tags:
        logger.error("No tags were generated. Cannot proceed with tag assignment.")
        return
    
    # Assign tags to the selected entries only
    logger.info(f"Assigning tags to {len(entries_to_tag)} selected entries...")
    tagged_entries = tagger.assign_tags_to_abstracts(entries_to_tag, tags)
    
    # Add entries without abstracts (with empty tags)
    entries_without_abstracts = [e for e in entries if not e.get('abstract')]
    for entry in entries_without_abstracts:
        entry['tags'] = ''
    
    # Combine all entries
    all_entries = tagged_entries + entries_without_abstracts
    
    # Save results
    df = pd.DataFrame(all_entries)
    df.to_csv(output_path, index=False)
    
    # Log summary
    logger.info(f"Saved {len(all_entries)} entries to {output_path}")
    logger.info(f"  - Entries with tags: {len(tagged_entries)}")
    logger.info(f"  - Entries without abstracts: {len(entries_without_abstracts)}")
    logger.info(f"Tag generation used {tag_sample_size} samples from {len(entries_to_tag)} processed entries")
    
    # Generate word cloud if we have tags
    if not df.empty and 'tags' in df.columns and not df['tags'].isna().all():
        try:
            # Flatten all tags and count frequencies
            all_tags = []
            for tag_list in df['tags']:
                if pd.notna(tag_list):
                    tags = [tag.strip().lower() for tag in str(tag_list).split(",")]
                    all_tags.extend(tags)
            
            tag_counts = {}
            for tag in all_tags:
                tag = tag.strip()
                if tag and tag.lower() != 'nan':  # Skip empty and 'nan' tags
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            if not tag_counts:
                logger.warning("No valid tags found for word cloud generation")
                return
                
            # Skip word cloud generation if not requested
            if not hasattr(args, 'wordcloud') or not args.wordcloud:
                return
                
            # Generate word cloud in the requested format(s)
            base_path = output_path.with_suffix('')
            
            if args.wordcloud in ['png', 'both']:
                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt
                    
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=1200,
                        height=800,
                        background_color='white',
                        max_words=100,
                        colormap='viridis',
                        prefer_horizontal=0.9,
                        scale=2,
                        min_font_size=10,
                        max_font_size=120
                    ).generate_from_frequencies(tag_counts)
                    
                    # Save the word cloud
                    png_path = f"{base_path}_wordcloud.png"
                    plt.figure(figsize=(12, 8), dpi=100)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    plt.savefig(png_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    logger.info(f"PNG word cloud saved to {png_path}")
                except Exception as e:
                    logger.error(f"Error generating PNG word cloud: {e}")
            
            if args.wordcloud in ['html', 'both']:
                try:
                    from .visualization import create_interactive_wordcloud
                    
                    # Create interactive word cloud
                    html_path = f"{base_path}_wordcloud.html"
                    create_interactive_wordcloud(
                        df,
                        output_file=html_path,
                        tag_column='tags',
                        title=f'Interactive Word Cloud - {output_path.stem}'
                    )
                    logger.info(f"Interactive word cloud saved to {html_path}")
                except Exception as e:
                    logger.error(f"Error generating interactive word cloud: {e}")
                    
        except Exception as e:
            logger.error(f"Error generating word cloud: {e}")

def visualize_command(args: argparse.Namespace) -> None:
    """Handle the visualize command.
    
    Args:
        args: Parsed command line arguments
    """
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Generate visualizations
    if args.interactive or (not args.static and not args.network):
        logger.info("Generating interactive word cloud...")
        fig = create_plotly_wordcloud(df)
        fig.write_html(output_dir / "wordcloud_interactive.html")
        
        logger.info("Generating interactive tag cloud...")
        from .visualization.wordcloud_plotly import create_interactive_plot
        fig = create_interactive_plot(df)
        fig.write_html(output_dir / "tag_cloud_interactive.html")
    
    if args.static or (not args.interactive and not args.network):
        logger.info("Generating static word cloud...")
        create_mpl_wordcloud(
            df,
            output_file=output_dir / "wordcloud.png",
            show=False
        )
        
        logger.info("Generating tag frequencies plot...")
        plot_tag_frequencies(
            df,
            output_file=output_dir / "tag_frequencies.png",
            show=False
        )
    
    if args.network:
        logger.info("Generating tag co-occurrence network...")
        from .visualization.interactive_plot import create_tag_network
        fig = create_tag_network(df)
        if fig:
            fig.write_html(output_dir / "tag_network.html")
    
    logger.info(f"Visualizations saved to {output_dir}")

def dashboard_command(args: argparse.Namespace) -> None:
    """Handle the dashboard command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.express as px
        import pandas as pd
        
        # Load data if a default file exists
        data_file = "tagged_abstracts.csv"
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            has_data = True
        else:
            df = pd.DataFrame()
            has_data = False
        
        # Initialize the Dash app
        app = dash.Dash(__name__)
        
        # Define the layout
        app.layout = html.Div([
            html.H1("Bibtex Analyzer Dashboard"),
            
            # File upload section
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False
            ),
            
            # Main content
            html.Div([
                # Left panel
                html.Div([
                    html.H3("Tag Cloud"),
                    dcc.Graph(id='wordcloud-plot')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                
                # Right panel
                html.Div([
                    html.H3("Tag Frequencies"),
                    dcc.Graph(id='frequencies-plot')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
            ]),
            
            # Hidden div to store the data
            html.Div(id='data-store', style={'display': 'none'})
        ])
        
        # Callbacks would go here in a real implementation
        
        # Run the app
        app.run_server(debug=args.debug, port=args.port)
        
    except ImportError as e:
        logger.error("Dashboard dependencies not installed. Install with 'pip install dash pandas plotly'")
        sys.exit(1)

def main() -> None:
    """Main entry point for the Bibtex Analyzer CLI."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args(sys.argv[1:])
    
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(log_level)
    
    # Execute the appropriate command
    if args.command == "analyze":
        analyze_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    elif args.command == "dashboard":
        dashboard_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
