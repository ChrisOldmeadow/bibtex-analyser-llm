"""Interactive dashboard for Bibtex Analyzer."""
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
from dash.exceptions import PreventUpdate
from datetime import datetime
import matplotlib

# Use the 'Agg' backend to avoid GUI threading issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .bibtex_processor import process_bibtex_file
from .tag_generator import TagGenerator

# Initialize the Dash app with Bootstrap theme
def create_dashboard(debug: bool = False, port: int = 8050) -> dash.Dash:
    """Create and configure the Dash application.
    
    Args:
        debug: Whether to run in debug mode
        port: Port to run the dashboard on
        
    Returns:
        Configured Dash application
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Bibtex Analyzer Dashboard",
        update_title=None
    )
    
    # Create upload directory if it doesn't exist
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Add initial store for tracking first load
    app.layout = html.Div([
        # Stores
        dcc.Store(id='initial-load', data=True),
        dcc.Store(id='data-store'),
        dcc.Store(id='tagged-data-store'),
        dcc.Store(id='wordcloud-store'),
        dcc.Store(id='wordcloud-html-store', data=None),
        dcc.Download(id="download-wordcloud"),
        dcc.Download(id="download-tagged-data"),
        
        # Navigation bar
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/yourusername/bibtex-analyzer-gpt")),
            ],
            brand="Bibtex Analyzer Dashboard",
            brand_href="#",
            color="primary",
            dark=True,
            fluid=True,
        ),
        
        # Main content
        dbc.Container(fluid=True, className="py-4", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("1. Upload & Process BibTeX File"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-bibtex',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select BibTeX File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0',
                                    'cursor': 'pointer'
                                },
                                multiple=False
                            ),
                            html.Div(id='upload-filename'),
                            html.Div(id='upload-status', className='mt-2'),
                            
                            # Processing options
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Tag Sample Size", html_for="tag-sample-size"),
                                    dbc.Input(
                                        id="tag-sample-size",
                                        type="number",
                                        min=1,
                                        value=30,
                                        className="mb-3"
                                    ),
                                    dbc.FormText("Number of entries to use for tag generation"),
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Max Entries to Tag", html_for="max-entries-to-tag"),
                                    dbc.Input(
                                        id="max-entries-to-tag",
                                        type="number",
                                        min=0,
                                        value=0,
                                        className="mb-3"
                                    ),
                                    dbc.FormText("Max entries to tag (0 for all)"),
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Model", html_for="model-select"),
                                    dbc.Select(
                                        id="model-select",
                                        options=[
                                            {"label": "GPT-3.5 Turbo", "value": "gpt-3.5-turbo"},
                                            {"label": "GPT-4", "value": "gpt-4"},
                                        ],
                                        value="gpt-3.5-turbo",
                                        className="mb-3"
                                    ),
                                ], width=4),
                            ]),
                            
                            dbc.Button(
                                "Process File",
                                id="process-button",
                                color="primary",
                                className="w-100",
                                disabled=True
                            ),
                            dbc.Progress(id="process-progress", style={"height": "5px"}, className="mt-2"),
                            html.Div(id="process-progress-text", className="text-center small text-muted mt-1"),
                            # Log display area
                            dbc.Card([
                                dbc.CardHeader("Processing Logs"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-logs",
                                        type="circle",
                                        children=html.Div(
                                            id="processing-logs", 
                                            style={
                                                "maxHeight": "200px", 
                                                "overflowY": "auto", 
                                                "whiteSpace": "pre-wrap",
                                                "fontFamily": "monospace",
                                                "fontSize": "0.8rem",
                                                "backgroundColor": "#f8f9fa",
                                                "padding": "10px",
                                                "borderRadius": "5px"
                                            }
                                        )
                                    )
                                ])
                            ], className="mt-3"),
                        ]),
                    ], className="mb-4"),
                    
                    # Word cloud customization
                    dbc.Card([
                        dbc.CardHeader("3. Customize Word Cloud"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Max Words"),
                                    dcc.Slider(
                                        id="max-words-slider",
                                        min=10,
                                        max=200,
                                        step=10,
                                        value=100,
                                        marks={i: str(i) for i in range(0, 201, 50)},
                                        className="mb-3"
                                    ),
                                ], width=12),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Color Scheme"),
                                    dcc.Dropdown(
                                        id="color-scheme",
                                        options=[
                                            {"label": "Viridis", "value": "viridis"},
                                            {"label": "Plasma", "value": "plasma"},
                                            {"label": "Inferno", "value": "inferno"},
                                            {"label": "Magma", "value": "magma"},
                                            {"label": "Cividis", "value": "cividis"},
                                            {"label": "Rainbow", "value": "rainbow"},
                                        ],
                                        value="viridis",
                                        className="mb-3"
                                    ),
                                ], width=6),
                                
                                dbc.Col([
                                    dbc.Label("Background Color"),
                                    dcc.Dropdown(
                                        id="bg-color",
                                        options=[
                                            {"label": "White", "value": "white"},
                                            {"label": "Black", "value": "black"},
                                            {"label": "Light Gray", "value": "#f8f9fa"},
                                            {"label": "Dark", "value": "#212529"},
                                        ],
                                        value="white",
                                        className="mb-3"
                                    ),
                                ], width=6),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Word Cloud Type"),
                                    dbc.RadioItems(
                                        id="wordcloud-type",
                                        options=[
                                            {"label": "Static", "value": "static"},
                                            {"label": "Interactive", "value": "interactive"},
                                        ],
                                        value="static",
                                        inline=True,
                                        className="mb-3"
                                    ),
                                ], width=12),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Update Word Cloud",
                                        id="generate-wc-button",
                                        color="primary",
                                        className="w-100 mt-2",
                                        disabled=False,
                                        n_clicks=0
                                    ),
                                ], width=12),
                            ]),
                        ]),
                    ], className="mb-4"),
                ], md=4),
                
                # Visualization area
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Word Cloud", tab_id="wordcloud-tab", children=[
                            html.Div(className="text-center mt-3", children=[
                                dcc.Graph(
                                    id='word-cloud-graph',
                                    config={'displayModeBar': False},
                                    style={
                                        'width': '100%',
                                        'height': '600px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                        'display': 'none'
                                    }
                                ),
                                html.Div(id="selected-word", className="h4 mt-3 mb-2"),
                                dbc.Alert(
                                    id="wordcloud-error",
                                    color="danger",
                                    is_open=False,
                                    dismissable=True,
                                    className="mt-3"
                                ),
                                html.Div(id="papers-container", className="mt-3")
                            ]),
                            dbc.Row(
                                dbc.Col([
                                    dbc.Button(
                                        "Download Word Cloud",
                                        id="download-wc-button",
                                        color="success",
                                        className="mt-3 w-100",
                                        disabled=True
                                    ),
                                ], width=6, className="mx-auto"),
                                className="mb-3"
                            ),
                        ]),
                        
                        dbc.Tab(label="Tag Frequencies", tab_id="frequencies-tab", children=[
                            dcc.Graph(id="frequencies-plot", className="mt-3"),
                            dbc.Row(
                                dbc.Col([
                                    dbc.Button(
                                        "Download Frequencies",
                                        id="download-freq-button",
                                        color="success",
                                        className="mt-3 w-100"
                                    ),
                                ], width=6, className="mx-auto"),
                                className="mb-3"
                            ),
                        ]),
                        
                        dbc.Tab(label="Data Table", tab_id="table-tab", children=[
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Filter by Tags", html_for="tag-filter"),
                                    dcc.Dropdown(
                                        id="tag-filter",
                                        options=[],
                                        multi=True,
                                        searchable=True,
                                        placeholder="Select tags to filter...",
                                        className="mb-3"
                                    ),
                                ], width=4),
                                dbc.Col([
                                    dbc.RadioItems(
                                        id="download-type",
                                        options=[
                                            {"label": "Download All Entries", "value": "all"},
                                            {"label": "Download Filtered Entries", "value": "filtered"}
                                        ],
                                        value="all",
                                        inline=True,
                                        className="mb-2"
                                    ),
                                    dbc.Button(
                                        "Download BibTeX",
                                        id="download-bibtex-btn",
                                        color="primary",
                                        className="mb-3"
                                    )
                                ], width=4),
                            ]),
                            html.Div(id="data-table-container", className="mt-3"),
                        ]),
                    ], id="tabs", active_tab="wordcloud-tab"),
                ], md=8),
            ]),
        ]),
        
        # Footer
        dbc.Navbar(
            dbc.Container([
                html.Div(
                    [
                        html.Span(" 2023 Bibtex Analyzer ", className="text-muted"),
                        html.A("GitHub", href="https://github.com/yourusername/bibtex-analyzer-gpt", className="text-muted"),
                    ],
                    className="text-center w-100"
                )
            ]),
            color="light",
            className="fixed-bottom py-2"
        ),
    ])
    
    # Register callbacks
    register_callbacks(app, UPLOAD_DIR)
    
    return app

# Utility functions for dashboard callbacks

def create_logger() -> callable:
    """Create a logger function for tracking processing steps.
    
    Returns:
        A function that updates logs with timestamps
    """
    logs = []
    
    def update_log(message: str) -> str:
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        logs.append(log_entry)
        return "\n".join(logs[-20:])  # Keep last 20 log entries
    
    return update_log

def count_tag_frequencies(df: pd.DataFrame) -> Dict[str, int]:
    """Count the frequency of each tag in the DataFrame.
    
    Args:
        df: DataFrame containing a 'tags' column
        
    Returns:
        Dictionary mapping tags to their frequencies
    """
    tag_counts = {}
    for tags in df['tags'].dropna():
        if isinstance(tags, str):
            for tag in [t.strip() for t in tags.split(',')]:
                if tag and tag.lower() != 'nan':
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return tag_counts

def extract_unique_tags(df: pd.DataFrame) -> set:
    """Extract unique tags from the DataFrame.
    
    Args:
        df: DataFrame containing a 'tags' column
        
    Returns:
        Set of unique tags
    """
    all_tags = set()
    for tags_str in df['tags'].dropna():
        if isinstance(tags_str, str):
            tags = [tag.strip() for tag in tags_str.split(',')]
            all_tags.update(tags)
    return all_tags

def create_paper_cards(papers: List[Dict]) -> List[dbc.Card]:
    """Create Dash cards for papers.
    
    Args:
        papers: List of paper data dictionaries
        
    Returns:
        List of Dash Card components
    """
    paper_cards = []
    for paper in papers:
        card = dbc.Card([
            dbc.CardHeader(f"{paper['year']}", className="text-muted"),
            dbc.CardBody([
                html.H5(paper['title'], className="card-title"),
                html.P(f"{paper['authors']}", className="card-text"),
                dbc.Button("View Paper", href=paper['url'], color="primary", 
                         external_link=True, target="_blank") 
                if paper['url'] and paper['url'] != '#' else None
            ])
        ], className="mb-3")
        paper_cards.append(card)
    return paper_cards

def df_from_json_store(df_json: Optional[str]) -> Optional[pd.DataFrame]:
    """Convert stored JSON data to DataFrame.
    
    Args:
        df_json: JSON string representing DataFrame
        
    Returns:
        DataFrame or None if conversion fails
    """
    if not df_json:
        return None
    try:
        return pd.read_json(io.StringIO(df_json), orient='split')
    except Exception:
        return None

def find_papers_with_tag(df: pd.DataFrame, tag: str) -> List[Dict]:
    """Find papers containing a specific tag.
    
    Args:
        df: DataFrame with papers data
        tag: Tag to search for
        
    Returns:
        List of dictionaries with paper information
    """
    papers = []
    for idx, row in df.iterrows():
        if pd.notna(row['tags']):
            tags = [t.strip() for t in str(row['tags']).split(',')]
            if tag in tags:
                papers.append({
                    'title': row.get('title', 'No title'),
                    'year': row.get('year', 'N/A'),
                    'authors': row.get('author', 'Unknown').split(' and ')[0],
                    'journal': row.get('journal', ''),
                    'url': row.get('url', '#')
                })
    
    # Sort papers by year (newest first)
    papers.sort(key=lambda x: str(x['year']), reverse=True)
    return papers

def handle_upload_status(contents: Optional[str], filename: Optional[str], upload_dir: Path) -> Tuple[str, bool, Optional[dbc.Alert]]:
    """Update the upload status and enable/disable the process button.
    
    Args:
        contents: File contents
        filename: Name of the uploaded file
        upload_dir: Directory to store uploads
        
    Returns:
        Tuple containing filename display text, button disabled state, and status alert
    """
    if contents is None:
        raise PreventUpdate
        
    # Clear any previous uploads
    for f in upload_dir.glob("*"):
        f.unlink()
        
    # Save the uploaded file
    if filename and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save the file
        file_path = upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(decoded)
            
        return (
            f"Selected: {filename}",
            False,
            dbc.Alert(f"File '{filename}' uploaded successfully!", color="success")
        )
    
    return "", True, ""

def handle_process_bibtex(n_clicks: Optional[int], filename: Optional[str], 
                         tag_sample_size: int, max_entries_to_tag: int,
                         model: str, upload_dir: Path) -> Tuple[Union[str, Any], Union[str, Any], 
                                                             int, str, Optional[dbc.Alert], str]:
    """Process the uploaded BibTeX file and generate tags with detailed logging.
    
    Args:
        n_clicks: Number of button clicks
        filename: Name of the uploaded file
        tag_sample_size: Sample size for tag generation
        max_entries_to_tag: Maximum entries to tag
        model: Model to use for tag generation
        upload_dir: Directory where uploads are stored
        
    Returns:
        Tuple containing data store, tagged data store, progress value, progress label, 
        status alert, and processing logs
    """
    if n_clicks is None or not filename:
        raise PreventUpdate
        
    file_path = upload_dir / filename
    update_log = create_logger()
    
    if not file_path.exists():
        error_msg = "Error: File not found"
        return (
            no_update,
            no_update,
            0,
            error_msg,
            dbc.Alert(error_msg, color="danger"),
            update_log(error_msg)
        )
    
    try:
        # Process the BibTeX file
        try:
            log = update_log(f"Starting processing of {filename}")
            entries = process_bibtex_file(file_path)
            log = update_log(f"Found {len(entries)} entries in the BibTeX file")
            
            if not entries:
                error_msg = "Error: No entries found in the BibTeX file"
                return (
                    no_update,
                    no_update,
                    0,
                    error_msg,
                    dbc.Alert(error_msg, color="danger"),
                    update_log(error_msg)
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(entries)
            
            # Filter out entries without abstracts
            df = df[df['abstract'].notna() & (df['abstract'] != '')]
            log = update_log(f"Filtered to {len(df)} entries with abstracts")
            
            if df.empty:
                error_msg = "Error: No entries with abstracts found"
                return (
                    no_update,
                    no_update,
                    0,
                    error_msg,
                    dbc.Alert(error_msg, color="danger"),
                    update_log(error_msg)
                )
                
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            return (
                no_update,
                no_update,
                0,
                error_msg,
                dbc.Alert(error_msg, color="danger"),
                update_log(error_msg)
            )
        
        # Generate tags
        log = update_log(f"Initializing tag generator with model: {model}")
        tag_generator = TagGenerator(model=model)
        
        # Update progress
        progress = 0
        progress_interval = 100 / min(tag_sample_size, len(df))
        
        # Generate tags for a subset of entries
        if tag_sample_size > 0:
            sample_size = min(tag_sample_size, len(df))
            log = update_log(f"Generating tags for {sample_size} entries...")
            
            df_sample = df.sample(sample_size)
            entries = df_sample.to_dict('records')
            
            log = update_log("Starting tag generation...")
            try:
                tags = tag_generator.generate_tags_for_abstracts(entries)
                log = update_log(f"Generated {len(tags) if tags else 0} unique tags")
                
                if tags:
                    log = update_log("Sample tags: " + ", ".join(list(tags)[:10]) + ("..." if len(tags) > 10 else ""))
                
                # Assign tags using the TagGenerator's method (same as CLI)
                log = update_log("Assigning tags to entries using TagGenerator...")
                
                # Convert entries to list of dicts if needed
                entries_list = df.to_dict('records')
                
                # Filter out entries without abstracts
                entries_with_abstracts = [e for e in entries_list if e.get('abstract')]
                if len(entries_with_abstracts) < len(entries_list):
                    log = update_log(f"Skipped {len(entries_list) - len(entries_with_abstracts)} entries without abstracts")
                
                if not entries_with_abstracts:
                    log = update_log("No entries with abstracts found. Cannot assign tags.")
                else:
                    # Apply max entries limit if specified
                    if max_entries_to_tag and max_entries_to_tag > 0:
                        import random
                        max_entries = min(max_entries_to_tag, len(entries_with_abstracts))
                        entries_to_tag = random.sample(entries_with_abstracts, max_entries)
                        log = update_log(f"Randomly selected {max_entries} entries to tag (from {len(entries_with_abstracts)} with abstracts)")
                    else:
                        entries_to_tag = entries_with_abstracts
                        log = update_log(f"Tagging all {len(entries_to_tag)} entries with abstracts")
                    
                    # Use the TagGenerator's method to assign tags
                    log = update_log(f"Assigning tags to {len(entries_to_tag)} entries...")
                    tagged_entries = tag_generator.assign_tags_to_abstracts(entries_to_tag, tags)
                    
                    # Update the original dataframe with the tagged entries
                    df = pd.DataFrame(tagged_entries + 
                                   [e for e in entries_list if not e.get('abstract')])
                    
                    # Log the results
                    tagged_count = len([e for e in tagged_entries if e.get('tags')])
                    log = update_log(f"Successfully assigned tags to {tagged_count} out of {len(entries_with_abstracts)} entries with abstracts")
                    log = update_log(f"Total entries with tags: {df['tags'].count()}")
                    
                    # Show a sample of tagged entries
                    sample_size = min(3, len(tagged_entries))
                    if sample_size > 0:
                        sample = tagged_entries[:sample_size]
                        for entry in sample:
                            log = update_log(f"Sample - ID: {entry.get('ID', 'N/A')}, Tags: {entry.get('tags', 'None')}")
                
            except Exception as e:
                error_msg = f"Error during tag generation: {str(e)}"
                return (
                    no_update,
                    no_update,
                    progress,
                    f"Error: {str(e)}",
                    dbc.Alert(error_msg, color="danger"),
                    update_log(error_msg)
                )
        
        # Save the results
        log = update_log("Saving results...")
        df_json = df.to_json(orient='split', date_format='iso')
        
        log = update_log("Processing complete!")
        
        return (
            df_json,
            df_json,
            100,
            "Processing complete!",
            dbc.Alert("File processed successfully!", color="success"),
            log
        )
        
    except Exception as e:
        progress = 0 if 'progress' not in locals() else progress
        error_msg = f"Unexpected error: {str(e)}"
        return (
            no_update,
            no_update,
            progress,
            f"Error: {str(e)}",
            dbc.Alert(error_msg, color="danger"),
            update_log(error_msg)
        )

def handle_generate_word_cloud(n_clicks: Optional[int], initial_load_ts: Optional[int], 
                         df_json: Optional[str], max_words: int, color_scheme: str, 
                         bg_color: str, wc_type: str) -> Tuple:
    """Generate and display the word cloud using the wordcloud library.
    
    Args:
        n_clicks: Number of button clicks
        initial_load_ts: Initial load timestamp
        df_json: JSON string containing DataFrame
        max_words: Maximum number of words in cloud
        color_scheme: Color scheme for word cloud
        bg_color: Background color
        wc_type: Type of word cloud (static or interactive)
        
    Returns:
        Tuple containing figure, HTML data, button disabled state, error message, error visibility, and style
    """
    # Get the callback context
    ctx = dash.callback_context
    
    # Check if this is the initial load or triggered by initial load
    if not ctx.triggered:
        # On initial load, check if we have data
        if df_json is None:
            return go.Figure(), None, True, None, False, {'display': 'none'}
        # If we have data, generate a static word cloud
        wc_type = 'static'
        return handle_generate_word_cloud(None, initial_load_ts, df_json, max_words, color_scheme, bg_color, wc_type)
    
    # If triggered by initial load or data store update
    if 'initial-load' in ctx.triggered[0]['prop_id'] or 'tagged-data-store' in ctx.triggered[0]['prop_id']:
        if df_json is None:
            return go.Figure(), None, True, None, False, {'display': 'none'}
        wc_type = 'static'
    
    # If no data, show upload message
    elif df_json is None:
        return go.Figure(), None, True, "Please upload and process a BibTeX file first.", True, {'display': 'none'}
    
    try:
        # Get the data
        df = df_from_json_store(df_json)
        
        if df is None or 'tags' not in df.columns or df['tags'].isna().all():
            return go.Figure(), None, True, "No tags found in the data.", True, {'display': 'block'}
        
        # Get tag frequencies
        tag_counts = count_tag_frequencies(df)
        
        if not tag_counts:
            return go.Figure(), None, True, "No valid tags found.", True, {'display': 'block'}
        
        # Sort tags by frequency and get top N
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit the number of tags to max_words
        if max_words and max_words < len(sorted_tags):
            sorted_tags = sorted_tags[:max_words]
        
        # Convert to word:count dictionary for word cloud
        tag_counts_limited = dict(sorted_tags)
        
        # Create paper mappings
        word_to_papers = {}
        for idx, row in df.iterrows():
            if pd.notna(row['tags']):
                paper_tags = [tag.strip() for tag in row['tags'].split(',')]
                for tag in paper_tags:
                    if tag in tag_counts_limited:
                        if tag not in word_to_papers:
                            word_to_papers[tag] = []
                        word_to_papers[tag].append({
                            'title': row.get('title', 'No title'),
                            'year': row.get('year', 'N/A'),
                            'authors': row.get('author', 'Unknown').split(' and ')[0],
                            'url': row.get('url', '#')
                        })
        
        if not word_to_papers:
            return go.Figure(), None, True, "No papers found for tags.", True, {'display': 'block'}
        
        # Create a word cloud using the wordcloud library
        try:
            # Create a word cloud with specified parameters
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color=bg_color,
                max_words=max_words or len(tag_counts_limited),
                colormap=color_scheme,
                prefer_horizontal=0.9,
                scale=2,
                min_font_size=10,
                max_font_size=120,
                relative_scaling=0.5,
                collocations=False,
                normalize_plurals=False
            )
            
            # Generate word cloud from frequencies
            wordcloud.generate_from_frequencies(tag_counts_limited)
            
            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor=bg_color)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor=bg_color)
            plt.close()
            
            # Encode image
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            if wc_type == 'static':
                return generate_static_word_cloud(img_str, bg_color)
            else:  # Interactive mode
                return generate_interactive_word_cloud(tag_counts_limited, word_to_papers, bg_color, img_str)
                
        except Exception as e:
            import traceback
            print(f"Error generating word cloud: {str(e)}")
            print(traceback.format_exc())
            return go.Figure(), None, True, f"Error generating word cloud: {str(e)}", True, {'display': 'block'}
    except Exception as e:
        import traceback
        print(f"Error in word cloud generation: {str(e)}")
        print(traceback.format_exc())
        return go.Figure(), None, True, f"Error: {str(e)}", True, {'display': 'block'}

def generate_static_word_cloud(img_str: str, bg_color: str) -> Tuple:
    """Generate a static word cloud.
    
    Args:
        img_str: Base64 encoded image string
        bg_color: Background color
        
    Returns:
        Tuple containing figure and other word cloud outputs
    """
    # Create static figure with image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_str}',
            xref='paper',
            yref='paper',
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing='contain',
            opacity=1,
            layer='above'
        )
    )
    
    # Update layout for static mode
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1]
        ),
        showlegend=False,
        height=600,
        width=800
    )
    
    return fig, None, False, None, False, {
        'width': '100%',
        'height': '600px',
        'border': 'none',
        'borderRadius': '5px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'display': 'block'
    }

def generate_interactive_word_cloud(tag_counts_limited: Dict[str, int], 
                                   word_to_papers: Dict[str, List[Dict]], 
                                   bg_color: str, img_str: str) -> Tuple:
    """Generate an interactive word cloud.
    
    Args:
        tag_counts_limited: Dictionary of tags and their frequencies
        word_to_papers: Dictionary mapping words to papers
        bg_color: Background color
        img_str: Base64 encoded image string
        
    Returns:
        Tuple containing figure and other word cloud outputs
    """
    # Store HTML data for click handling
    html_data = {
        'words': list(tag_counts_limited.keys()),
        'frequencies': list(tag_counts_limited.values()),
        'papers': word_to_papers,
        'img_data': img_str
    }
    
    # Create scatter plot with random positioning and frequency-based sizing
    words = list(tag_counts_limited.keys())
    x = [random.random() for _ in words]
    y = [random.random() for _ in words]
    sizes = [min(100, max(10, freq * 2)) for freq in tag_counts_limited.values()]
    
    # Create hover text with word frequency and related papers
    hover_texts = []
    for word in words:
        papers = word_to_papers.get(word, [])
        hover_text = f"<b>{word}</b><br>Frequency: {tag_counts_limited[word]}"
        if papers:
            hover_text += "<br><br>Related Papers:<br>" + \
                "<br>".join([f"{p['year']}: {p['title']}" for p in papers[:3]])
        hover_texts.append(hover_text)
    
    # Create scatter plot with all words
    fig = go.Figure()
    
    # Create a trace for each word with its own size
    for word, freq in tag_counts_limited.items():
        word_x = x[words.index(word)]
        word_y = y[words.index(word)]
        
        # Calculate font size based on frequency (scaled to be visually appealing)
        base_size = 10  # Base size for smallest words
        max_size = 50   # Maximum size for most frequent words
        font_size = base_size + (max_size - base_size) * (freq / max(tag_counts_limited.values()))
        
        fig.add_trace(go.Scatter(
            x=[word_x],
            y=[word_y],
            mode='text',
            text=[word],
            textfont=dict(
                size=font_size,
                color='rgba(0,0,0,0.7)'
            ),
            hoverinfo='text',
            hovertext=[hover_texts[words.index(word)]]
        ))
    
    # Update layout for interactive mode
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1]
        ),
        showlegend=False,
        height=600,
        width=800,
        clickmode='event+select'
    )
    
    return fig, html_data, False, None, False, {
        'width': '100%',
        'height': '600px',
        'border': 'none',
        'borderRadius': '5px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'display': 'block'
    }

def handle_update_frequencies_plot(df_json: Optional[str], color_scheme: str) -> go.Figure:
    """Update the tag frequencies plot.
    
    Args:
        df_json: JSON string containing DataFrame
        color_scheme: Color scheme for the plot
        
    Returns:
        Plotly figure object
    """
    try:
        if df_json is None:
            return go.Figure()
            
        # Load the data
        df = df_from_json_store(df_json)
        if df is None:
            return go.Figure()
        
        # Count tag frequencies
        tag_counts = count_tag_frequencies(df)
        
        # Convert to DataFrame and sort
        df_tags = pd.DataFrame(tag_counts.items(), columns=['tag', 'count'])
        df_tags = df_tags.sort_values('count', ascending=False).head(20)
        
        # Create the bar chart
        fig = px.bar(
            df_tags,
            x='count',
            y='tag',
            orientation='h',
            color='count',
            color_continuous_scale=color_scheme,
            labels={'count': 'Number of Papers', 'tag': 'Tag'},
            title='Most Common Tags'
        )
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='white',
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=100, r=50, t=50, b=50),
            height=600
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in handle_update_frequencies_plot: {str(e)}")
        return go.Figure()

def handle_update_papers_display(clickData: Optional[Dict], df_json: Optional[str]) -> Tuple[List, str]:
    """Handle word cloud clicks and update the papers display.
    
    Args:
        clickData: Click data from the word cloud
        df_json: JSON string containing DataFrame
        
    Returns:
        Tuple containing paper cards and header text
    """
    # Debug information
    print(f"Callback triggered for handle_update_papers_display")
    
    # If no data, return empty
    if df_json is None:
        print("No data available")
        return [], ""
    
    # If triggered by word cloud click
    if clickData and clickData.get('points'):
        try:
            # Get the clicked word
            point = clickData['points'][0]
            word = point.get('customdata')
            if not word and 'text' in point:
                word = point['text']
            
            print(f"Clicked on word: {word}")
            
            if not word:
                print("No word found in click data")
                return [], ""
                
            # Load the data
            df = df_from_json_store(df_json)
            if df is None:
                return [], ""
            
            if 'tags' not in df.columns or df['tags'].isna().all():
                print("No tags found in data")
                return [html.Div("No tags found in the data.", className="text-muted")], ""
            
            # Find papers containing the clicked tag
            papers = find_papers_with_tag(df, word)
            
            print(f"Found {len(papers)} papers for tag: {word}")
            
            if not papers:
                print("No papers found for the selected tag")
                return [html.Div(f"No papers found with tag: {word}", 
                              className="text-muted")], f"Tag: {word}"
            
            # Create paper cards
            paper_cards = create_paper_cards(papers)
            
            return paper_cards, f"Papers with tag: {word}"
            
        except Exception as e:
            import traceback
            print(f"Error in handle_update_papers_display: {str(e)}")
            print(traceback.format_exc())
            return [html.Div(f"Error: {str(e)}", className="text-danger")], ""
    
    # Default return
    print("No matching condition, returning empty")
    return [], ""

def handle_update_data_table(selected_tags: List[str], df_json: Optional[str]) -> List[Dict]:
    """Update the data table with tag filtering.
    
    Args:
        selected_tags: List of selected tags to filter by
        df_json: JSON string containing DataFrame
        
    Returns:
        List of dictionary records for the data table
    """
    if not df_json:
        raise PreventUpdate
        
    df = df_from_json_store(df_json)
    
    if df is None or df.empty:
        raise PreventUpdate
    
    # Apply tag filter
    if selected_tags:
        # Convert tags to lowercase for case-insensitive matching
        selected_tags = [tag.lower().strip() for tag in selected_tags]
        
        # Split tags in each row and check if any match
        def has_selected_tags(tags_str):
            if not isinstance(tags_str, str):
                return False
            row_tags = [tag.lower().strip() for tag in tags_str.split(',')]
            return any(tag in row_tags for tag in selected_tags)
        
        mask = df['tags'].apply(has_selected_tags)
        df = df[mask]
    
    return df.to_dict('records')

def handle_download_bibtex_file(n_clicks: Optional[int], table_data: List[Dict], 
                              df_json: Optional[str], download_type: str) -> Dict:
    """Download the BibTeX file with tags.
    
    Args:
        n_clicks: Number of button clicks
        table_data: Data from the filtered table
        df_json: JSON string containing DataFrame
        download_type: Type of download (all or filtered)
        
    Returns:
        Dictionary with content and filename for download
    """
    if not df_json:
        raise PreventUpdate
        
    df = df_from_json_store(df_json)
    if df is None:
        raise PreventUpdate
    
    # Create BibTeX content with tags
    bibtex_content = """
@comment{BibTeX file generated by Bibtex Analyzer}
@comment{Tags have been added as a new field}
@comment{Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "}

"""
    
    # Define all possible BibTeX fields we want to include
    bibtex_fields = [
        'title', 'author', 'journal', 'year', 'volume', 'number', 'pages',
        'doi', 'url', 'abstract', 'keywords', 'month', 'publisher',
        'note', 'booktitle', 'chapter', 'edition', 'editor', 'institution',
        'organization', 'school', 'series', 'type', 'address', 'annote',
        'crossref', 'howpublished', 'key', 'organization', 'school',
        'issue', 'eprint', 'archivePrefix', 'primaryClass',
        'isbn', 'issn', 'language', 'location', 'day',
        'chapter', 'edition', 'series', 'type',
        'institution', 'organization', 'publisher',
        'school', 'address', 'month', 'day',
        'note', 'annote', 'keywords', 'abstract'
    ]
    
    # Get the data based on download type
    if download_type == "filtered":
        # Use the filtered data from the table
        df = pd.DataFrame(table_data)
    else:
        # Use all data from the tagged data store
        df = df_from_json_store(df_json)
        if df is None:
            raise PreventUpdate

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        # Get the entry type and ID
        entry_type = row.get('entry_type', 'article')
        entry_id = row.get('id', row.get('ID', f"entry_{_}"))  # Fallback to row index if no ID
        
        # Handle the case where ID might be in the entry field
        if 'entry' in row and isinstance(row['entry'], str) and '{' in row['entry']:
            try:
                entry_id = row['entry'].split('{')[1].split(',')[0].strip()
            except (IndexError, AttributeError):
                pass  # Keep the previously assigned entry_id
        
        # Start building the BibTeX entry
        bibtex_entry = f"@{entry_type}{{{entry_id},\n"
        
        # Add all available fields
        for field in bibtex_fields:
            if field in row and pd.notna(row[field]):
                value = str(row[field]).strip()
                # Escape special characters
                value = value.replace('{', '{{').replace('}', '}}').replace('"', '\"')
                # Format the field
                bibtex_entry += f"  {field} = {{\'{value}\'}},\n"
        
        # Add tags as keywords if they exist
        if 'tags' in row and pd.notna(row['tags']):
            tags = row['tags']
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
                bibtex_entry += f"  keywords = {{\'{', '.join(tags)}\'}},\n"
        
        # Add a newline and closing brace
        bibtex_entry += "}\n\n"
        
        # Validate the entry format
        if not bibtex_entry.startswith('@') or not bibtex_entry.endswith("}\n\n"):
            print(f"Warning: Invalid BibTeX entry format for ID: {entry_id}")
            continue
        
        bibtex_content += bibtex_entry
    
    # Add a summary comment at the end
    bibtex_content += f"@comment{{Total number of entries: {len(df)}}}\n"
    
    return dict(
        content=bibtex_content,
        filename="tagged_bibtex.bibtex"
    )

def handle_update_table_and_filter(df_json: Optional[str]) -> Tuple[Union[dash_table.DataTable, html.Div], List[Dict[str, str]]]:
    """Update the data table and populate tag filter options.
    
    Args:
        df_json: JSON string containing DataFrame
        
    Returns:
        Tuple containing data table component and tag filter options
    """
    if not df_json:
        return html.Div("No data available.", className="text-muted"), []
        
    df = df_from_json_store(df_json)
    
    if df is None or df.empty:
        return html.Div("No data available.", className="text-muted"), []
    
    # Get unique tags for filter options
    all_tags = extract_unique_tags(df)
    tag_options = [{'label': tag, 'value': tag} for tag in sorted(all_tags)]
    
    # Select only the available columns from our desired set
    columns_to_show = ['year', 'title', 'authors', 'journal', 'doi']
    available_columns = [col for col in columns_to_show if col in df.columns]
    
    # Format the display for available columns
    if 'authors' in df.columns:
        df['authors'] = df['authors'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
    
    df = df[available_columns]
    
    # Create the data table
    table = dash_table.DataTable(
        id='data-table',
        columns=[{"name": i, "id": i} for i in available_columns],
        data=df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '100px',
            'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return table, tag_options

def handle_download_word_cloud(n_clicks: Optional[int], img_data: Optional[str], 
                             html_data: Optional[Dict], wc_type: str) -> Optional[Union[Dict, Any]]:
    """Download the word cloud.
    
    Args:
        n_clicks: Number of button clicks
        img_data: Base64 encoded image data
        html_data: HTML data for the word cloud
        wc_type: Type of word cloud to download (png or html)
        
    Returns:
        Dictionary with content and filename or send_bytes function
    """
    if n_clicks is None:
        raise PreventUpdate
        
    if wc_type == 'png' and img_data:
        return dcc.send_bytes(
            lambda: base64.b64decode(img_data),
            "bibtex_wordcloud.png"
        )
    elif wc_type == 'html' and html_data:
        return dict(content=html_data, filename="bibtex_wordcloud.html")
    
    raise PreventUpdate

def register_callbacks(app: dash.Dash, upload_dir: Path) -> None:
    """Register all Dash callbacks.
    
    Args:
        app: Dash application
        upload_dir: Directory for uploaded files
    """
    @app.callback(
        [Output('upload-filename', 'children'),
         Output('process-button', 'disabled'),
         Output('upload-status', 'children')],
        [Input('upload-bibtex', 'contents')],
        [State('upload-bibtex', 'filename')]
    )
    def update_upload_status(contents, filename):
        """Update the upload status and enable/disable the process button."""
        return handle_upload_status(contents, filename, upload_dir)
    
    @app.callback(
        [Output('data-store', 'data'),
         Output('tagged-data-store', 'data'),
         Output('process-progress', 'value'),
         Output('process-progress', 'label'),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('processing-logs', 'children')],
        [Input('process-button', 'n_clicks')],
        [State('upload-bibtex', 'filename'),
         State('tag-sample-size', 'value'),
         State('max-entries-to-tag', 'value'),
         State('model-select', 'value')],
        prevent_initial_call=True
    )
    def process_bibtex(n_clicks, filename, tag_sample_size, max_entries_to_tag, model):
        """Process the uploaded BibTeX file and generate tags with detailed logging."""
        if n_clicks is None or not filename:
            raise PreventUpdate
            
        file_path = upload_dir / filename
        logs = []
        
        def update_log(message):
            timestamp = datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{timestamp}] {message}"
            logs.append(log_entry)
            return "\n".join(logs[-20:])  # Keep last 20 log entries
        
        if not file_path.exists():
            error_msg = "Error: File not found"
            return (
                no_update,
                no_update,
                0,
                error_msg,
                dbc.Alert(error_msg, color="danger"),
                update_log(error_msg)
            )
        
        try:
            # Process the BibTeX file
            try:
                log = update_log(f"Starting processing of {filename}")
                entries = process_bibtex_file(file_path)
                log = update_log(f"Found {len(entries)} entries in the BibTeX file")
                
                if not entries:
                    error_msg = "Error: No entries found in the BibTeX file"
                    return (
                        no_update,
                        no_update,
                        0,
                        error_msg,
                        dbc.Alert(error_msg, color="danger"),
                        update_log(error_msg)
                    )
                
                # Convert to DataFrame
                df = pd.DataFrame(entries)
                
                # Filter out entries without abstracts
                df = df[df['abstract'].notna() & (df['abstract'] != '')]
                log = update_log(f"Filtered to {len(df)} entries with abstracts")
                
                if df.empty:
                    error_msg = "Error: No entries with abstracts found"
                    return (
                        no_update,
                        no_update,
                        0,
                        error_msg,
                        dbc.Alert(error_msg, color="danger"),
                        update_log(error_msg)
                    )
                    
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                return (
                    no_update,
                    no_update,
                    0,
                    error_msg,
                    dbc.Alert(error_msg, color="danger"),
                    update_log(error_msg)
                )
            
            # Generate tags
            log = update_log(f"Initializing tag generator with model: {model}")
            tag_generator = TagGenerator(model=model)
            
            # Update progress
            progress = 0
            progress_interval = 100 / min(tag_sample_size, len(df))
            
            def progress_callback(progress_value, message=None):
                nonlocal progress
                progress = min(100, progress + progress_interval)
                if message:
                    log = update_log(message)
                return [
                    no_update,
                    no_update,
                    progress,
                    f"Processing: {int(progress)}%",
                    no_update,
                    log if 'log' in locals() else ""
                ]
            
            # Generate tags for a subset of entries
            if tag_sample_size > 0:
                sample_size = min(tag_sample_size, len(df))
                log = update_log(f"Generating tags for {sample_size} entries...")
                
                df_sample = df.sample(sample_size)
                entries = df_sample.to_dict('records')
                
                log = update_log("Starting tag generation...")
                try:
                    tags = tag_generator.generate_tags_for_abstracts(entries)
                    log = update_log(f"Generated {len(tags) if tags else 0} unique tags")
                    
                    if tags:
                        log = update_log("Sample tags: " + ", ".join(list(tags)[:10]) + ("..." if len(tags) > 10 else ""))
                    
                    # Assign tags using the TagGenerator's method (same as CLI)
                    log = update_log("Assigning tags to entries using TagGenerator...")
                    
                    # Convert entries to list of dicts if needed
                    entries_list = df.to_dict('records')
                    
                    # Filter out entries without abstracts
                    entries_with_abstracts = [e for e in entries_list if e.get('abstract')]
                    if len(entries_with_abstracts) < len(entries_list):
                        log = update_log(f"Skipped {len(entries_list) - len(entries_with_abstracts)} entries without abstracts")
                    
                    if not entries_with_abstracts:
                        log = update_log("No entries with abstracts found. Cannot assign tags.")
                    else:
                        # Apply max entries limit if specified
                        if max_entries_to_tag and max_entries_to_tag > 0:
                            import random
                            max_entries = min(max_entries_to_tag, len(entries_with_abstracts))
                            entries_to_tag = random.sample(entries_with_abstracts, max_entries)
                            log = update_log(f"Randomly selected {max_entries} entries to tag (from {len(entries_with_abstracts)} with abstracts)")
                        else:
                            entries_to_tag = entries_with_abstracts
                            log = update_log(f"Tagging all {len(entries_to_tag)} entries with abstracts")
                        
                        # Use the TagGenerator's method to assign tags
                        log = update_log(f"Assigning tags to {len(entries_to_tag)} entries...")
                        tagged_entries = tag_generator.assign_tags_to_abstracts(entries_to_tag, tags)
                        
                        # Update the original dataframe with the tagged entries
                        df = pd.DataFrame(tagged_entries + 
                                       [e for e in entries_list if not e.get('abstract')])
                        
                        # Log the results
                        tagged_count = len([e for e in tagged_entries if e.get('tags')])
                        log = update_log(f"Successfully assigned tags to {tagged_count} out of {len(entries_with_abstracts)} entries with abstracts")
                        log = update_log(f"Total entries with tags: {df['tags'].count()}")
                        
                        # Show a sample of tagged entries
                        sample_size = min(3, len(tagged_entries))
                        if sample_size > 0:
                            sample = tagged_entries[:sample_size]
                            for entry in sample:
                                log = update_log(f"Sample - ID: {entry.get('ID', 'N/A')}, Tags: {entry.get('tags', 'None')}")
                    
                except Exception as e:
                    error_msg = f"Error during tag generation: {str(e)}"
                    return (
                        no_update,
                        no_update,
                        progress,
                        f"Error: {str(e)}",
                        dbc.Alert(error_msg, color="danger"),
                        update_log(error_msg)
                    )
            
            # Save the results
            log = update_log("Saving results...")
            df_json = df.to_json(orient='split', date_format='iso')
            
            log = update_log("Processing complete!")
            
            return (
                df_json,
                df_json,
                100,
                "Processing complete!",
                dbc.Alert("File processed successfully!", color="success"),
                log
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            return (
                no_update,
                no_update,
                progress if 'progress' in locals() else 0,
                f"Error: {str(e)}",
                dbc.Alert(error_msg, color="danger"),
                update_log(error_msg)
            )
    
    @app.callback(
        [Output('word-cloud-graph', 'figure'),
         Output('wordcloud-html-store', 'data'),
         Output('download-wc-button', 'disabled'),
         Output('wordcloud-error', 'children'),
         Output('wordcloud-error', 'is_open'),
         Output('word-cloud-graph', 'style')],
        [Input('generate-wc-button', 'n_clicks'),
         Input('initial-load', 'modified_timestamp'),
         Input('tagged-data-store', 'data')],
        [State('max-words-slider', 'value'),
         State('color-scheme', 'value'),
         State('bg-color', 'value'),
         State('wordcloud-type', 'value')]
    )
    def generate_word_cloud(n_clicks, initial_load_ts, df_json, max_words, color_scheme, bg_color, wc_type):
        """Generate and display the word cloud using the wordcloud library."""
        return handle_generate_word_cloud(n_clicks, initial_load_ts, df_json, max_words, color_scheme, bg_color, wc_type)
    
    @app.callback(
        Output('frequencies-plot', 'figure'),
        [Input('tagged-data-store', 'data'),
         Input('color-scheme', 'value')]
    )
    def update_frequencies_plot(df_json, color_scheme):
        """Update the tag frequencies plot."""
        return handle_update_frequencies_plot(df_json, color_scheme)
            
    @app.callback(
        [Output('papers-container', 'children'),
         Output('selected-word', 'children')],
        [Input('word-cloud-graph', 'clickData'),
         Input('tagged-data-store', 'data')],
        prevent_initial_call=True
    )
    def update_papers_display(clickData, df_json):
        """Handle word cloud clicks and update the papers display."""
        return handle_update_papers_display(clickData, df_json)
    
    @app.callback(
        Output('data-table', 'data'),
        [Input('tag-filter', 'value')],
        [State('tagged-data-store', 'data')]
    )
    def update_data_table(selected_tags, df_json):
        """Update the data table with tag filtering."""
        return handle_update_data_table(selected_tags, df_json)
    
    @app.callback(
        Output("download-tagged-data", "data"),
        Input("download-bibtex-btn", "n_clicks"),
        State("data-table", "data"),
        State("tagged-data-store", "data"),
        State("download-type", "value"),
        prevent_initial_call=True
    )
    def download_bibtex_file(n_clicks, table_data, df_json, download_type):
        """Download the BibTeX file with tags."""
        return handle_download_bibtex_file(n_clicks, table_data, df_json, download_type)

    @app.callback(
        [Output('data-table-container', 'children'),
         Output('tag-filter', 'options')],
        [Input('tagged-data-store', 'data')],
        prevent_initial_call=True
    )
    def update_table_and_filter(df_json):
        """Update the data table and populate tag filter options."""
        return handle_update_table_and_filter(df_json)
    
    @app.callback(
        Output("download-wordcloud", "data"),
        [Input("download-wc-button", "n_clicks")],
        [State('wordcloud-store', 'data'),
         State('wordcloud-html-store', 'data'),
         State('wordcloud-type', 'value')],
        prevent_initial_call=True,
    )
    def download_word_cloud(n_clicks, img_data, html_data, wc_type):
        """Download the word cloud."""
        return handle_download_word_cloud(n_clicks, img_data, html_data, wc_type)
    


def run_dashboard(debug: bool = False, port: int = 8050) -> None:
    """Run the dashboard.
    
    Args:
        debug: Whether to run in debug mode
        port: Port to run the dashboard on
    """
    app = create_dashboard(debug=debug, port=port)
    app.run(debug=debug, port=port)

if __name__ == "__main__":
    run_dashboard(debug=True)
