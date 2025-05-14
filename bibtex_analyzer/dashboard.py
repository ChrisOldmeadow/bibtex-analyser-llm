"""Interactive dashboard for Bibtex Analyzer."""
import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
from dash.exceptions import PreventUpdate
import time
import numpy as np

from .bibtex_processor import process_bibtex_file
from .tag_generator import TagGenerator
from .visualization.wordcloud2 import create_interactive_wordcloud

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
        suppress_callback_exceptions=True
    )
    app.title = "Bibtex Analyzer Dashboard"
    
    # Create upload directory if it doesn't exist
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Define the layout
    app.layout = html.Div([
        dcc.Store(id='data-store'),
        dcc.Store(id='tagged-data-store'),
        dcc.Store(id='wordcloud-store'),
        dcc.Store(id='wordcloud-html-store'),
        dcc.Download(id="download-wordcloud"),
        dcc.Download(id="download-tagged-data"),
        
        # Navigation bar
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/yourusername/bibtex-analyzer-gpt")),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Documentation", href="#"),
                        dbc.DropdownMenuItem("About", href="#"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Help",
                ),
            ],
            brand="Bibtex Analyzer Dashboard",
            brand_href="#",
            color="primary",
            dark=True,
            fluid=True,
        ),
        
        # Main content
        dbc.Container(fluid=True, className="mt-4", children=[
            # File upload and processing section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("1. Upload BibTeX File"),
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
                        dbc.CardHeader("2. Customize Word Cloud"),
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
                                            {"label": "Static (PNG)", "value": "png"},
                                            {"label": "Interactive (HTML)", "value": "html"},
                                        ],
                                        value="png",
                                        inline=True,
                                        className="mb-3"
                                    ),
                                ], width=12),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Generate Word Cloud",
                                        id="generate-wc-button",
                                        color="primary",
                                        className="w-100",
                                        disabled=True
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
                            html.Div(id="wordcloud-container", className="text-center mt-3"),
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
                            html.Div(id="data-table-container", className="mt-3"),
                            dbc.Row(
                                dbc.Col([
                                    dbc.Button(
                                        "Download Data",
                                        id="download-data-button",
                                        color="success",
                                        className="mt-3 w-100"
                                    ),
                                ], width=6, className="mx-auto"),
                                className="mb-3"
                            ),
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
                        html.Span("© 2023 Bibtex Analyzer ", className="text-muted"),
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
        [Output('wordcloud-container', 'children'),
         Output('wordcloud-store', 'data'),
         Output('wordcloud-html-store', 'data'),
         Output('generate-wc-button', 'disabled'),
         Output('download-wc-button', 'disabled'),
         Output('processing-logs', 'children', allow_duplicate=True)],
        [Input('generate-wc-button', 'n_clicks'),
         Input('data-store', 'data'),
         Input('max-words-slider', 'value'),
         Input('color-scheme', 'value'),
         Input('bg-color', 'value'),
         Input('wordcloud-type', 'value')],
        prevent_initial_call=True
    )
    def generate_word_cloud(n_clicks, df_json, max_words, color_scheme, bg_color, wc_type):
        """Generate and display the word cloud using Plotly."""
        if df_json is None:
            return "Upload and process a BibTeX file first.", None, None, True, True, no_update
            
        try:
            # Get the data
            df = pd.read_json(io.StringIO(df_json), orient='split')
            
            if 'tags' not in df.columns or df['tags'].isna().all():
                return "No tags found in the data.", None, None, False, True, no_update
            
            # Flatten tags and count frequencies
            all_tags = []
            for tags in df['tags'].dropna():
                if isinstance(tags, str):
                    all_tags.extend([tag.strip() for tag in tags.split(',')])
            
            tag_counts = {}
            for tag in all_tags:
                tag = tag.strip()
                if tag and tag.lower() != 'nan':
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            if not tag_counts:
                return "No valid tags found.", None, None, False, True, no_update
            
            # Sort tags by frequency and get top N
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:max_words]
            words = [tag for tag, _ in sorted_tags]
            sizes = [count for _, count in sorted_tags]
            
            # Create a Plotly figure
            fig = go.Figure()
            
            # Get the color sequence based on the scheme
            try:
                color_seq = getattr(px.colors.sequential, color_scheme.upper())
                if not isinstance(color_seq, list):
                    color_seq = [color_seq]
                colors = color_seq * (len(words) // len(color_seq) + 1)  # Repeat colors if needed
            except (AttributeError, IndexError):
                colors = px.colors.sequential.Viridis  # Default color sequence
            
            # Calculate word sizes and positions
            max_size = max(sizes)
            word_elements = []
            
            # Create a grid for word placement
            grid_size = 100
            grid = [[False] * grid_size for _ in range(grid_size)]  # Grid to track occupied spaces
            
            # Function to check if a word can be placed at a position
            def can_place(x, y, width, height):
                x_start = max(0, int(x - width/2))
                x_end = min(grid_size-1, int(x + width/2))
                y_start = max(0, int(y - height/2))
                y_end = min(grid_size-1, int(y + height/2))
                
                for i in range(x_start, x_end + 1):
                    for j in range(y_start, y_end + 1):
                        if grid[i][j]:
                            return False
                return True
            
            # Function to mark grid as occupied
            def mark_occupied(x, y, width, height):
                x_start = max(0, int(x - width/2))
                x_end = min(grid_size-1, int(x + width/2))
                y_start = max(0, int(y - height/2))
                y_end = min(grid_size-1, int(y + height/2))
                
                for i in range(x_start, x_end + 1):
                    for j in range(y_start, y_end + 1):
                        grid[i][j] = True
            
            # Place words in a spiral pattern
            center_x, center_y = grid_size // 2, grid_size // 2
            angle_step = 0.2
            radius_step = 2
            
            for idx, (word, size) in enumerate(zip(words, sizes)):
                # Calculate font size based on word frequency
                font_size = min(10 + (size/max_size) * 40, 60)
                # Estimate word dimensions (rough approximation)
                word_width = len(word) * font_size * 0.4  # Approximate width
                word_height = font_size * 1.2  # Approximate height
                
                # Try to find a position for the word
                placed = False
                radius = 0
                angle = 0
                
                while not placed and radius < grid_size/2:
                    # Spiral coordinates
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        if can_place(x, y, word_width/10, word_height/10):
                            # Add word to the plot
                            word_elements.append({
                                'x': x/grid_size * 1000,  # Scale to plot size
                                'y': y/grid_size * 1000,
                                'text': word,
                                'size': font_size,
                                'color': colors[idx % len(colors)]
                            })
                            mark_occupied(x, y, word_width/10, word_height/10)
                            placed = True
                    
                    # Move along the spiral
                    angle += angle_step
                    if angle >= 2 * np.pi:
                        angle = 0
                        radius += radius_step
            
            # Add words to the plot
            for word_data in word_elements:
                fig.add_annotation(
                    x=word_data['x'],
                    y=word_data['y'],
                    text=word_data['text'],
                    showarrow=False,
                    font=dict(
                        size=word_data['size'],
                        color=word_data['color']
                    ),
                    xanchor='center',
                    yanchor='middle'
                )
            
            # Update layout for word cloud appearance
            fig.update_layout(
                plot_bgcolor=bg_color,
                paper_bgcolor=bg_color,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1000]),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1000]),
                showlegend=False,
                height=600,
                width=800,
            )
            
            # Convert figure to HTML
            html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # For PNG download, we'll use the HTML content
            # This is a limitation of the current implementation
            img_str = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            
            return (
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False},
                    style={
                        'width': '100%',
                        'height': '600px',
                        'border': 'none',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }
                ),
                img_str,
                html_content,
                False,
                False,
                no_update
            )
            
        except Exception as e:
            return f"Error generating word cloud: {str(e)}", None, None, False, True, no_update
    
    @app.callback(
        Output('frequencies-plot', 'figure'),
        [Input('tagged-data-store', 'data'),
         Input('color-scheme', 'value')]
    )
    def update_frequencies_plot(df_json, color_scheme):
        """Update the tag frequencies plot."""
        try:
            if df_json is None:
                return go.Figure()
                
            df = pd.read_json(io.StringIO(df_json), orient='split')
            
            if 'tags' not in df.columns or df['tags'].isna().all():
                return go.Figure()
            
            # Flatten tags and count frequencies
            all_tags = []
            for tags in df['tags'].dropna():
                if isinstance(tags, str):
                    all_tags.extend([tag.strip() for tag in tags.split(',')])
            
            tag_counts = {}
            for tag in all_tags:
                tag = tag.strip()
                if tag and tag.lower() != 'nan':
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Sort by frequency and get top 20
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            
            if not sorted_tags:
                return go.Figure()
            
            # Prepare data for plotting
            tags = [tag for tag, _ in sorted_tags]
            counts = [count for _, count in sorted_tags]
            
            # Create the bar chart with better error handling for color schemes
            try:
                # Try to use the selected color scheme, fallback to default if not available
                color_sequence = px.colors.sequential.get(color_scheme.capitalize(), 
                                                         px.colors.sequential.Viridis)
                
                # If we got a string (color name), convert it to a list
                if isinstance(color_sequence, str):
                    color_sequence = [color_sequence]
                    
                fig = px.bar(
                    x=tags,
                    y=counts,
                    labels={'x': 'Tag', 'y': 'Frequency'},
                    title='Top 20 Most Frequent Tags',
                    color=tags,
                    color_discrete_sequence=color_sequence
                )
            except Exception as e:
                # Fallback to default colors if there's any issue with the color scheme
                fig = px.bar(
                    x=tags,
                    y=counts,
                    labels={'x': 'Tag', 'y': 'Frequency'},
                    title='Top 20 Most Frequent Tags'
                )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=100),
                height=500,
                xaxis_title='',
                yaxis_title='Frequency',
                hovermode='closest'
            )
            
            # Improve bar appearance
            fig.update_traces(
                marker_line_width=1,
                marker_line_color='white',
                opacity=0.8
            )
            
            return fig
            
        except Exception as e:
            # Log the error and return an empty figure
            print(f"Error in update_frequencies_plot: {str(e)}")
            return go.Figure()
    
    @app.callback(
        Output('data-table-container', 'children'),
        [Input('tagged-data-store', 'data')]
    )
    def update_data_table(df_json):
        """Update the data table."""
        if df_json is None:
            return "No data available."
            
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        if df.empty:
            return "No data available."
        
        # Limit the number of rows for performance
        df_display = df.head(100).copy()
        
        # Format the display
        if 'authors' in df_display.columns:
            df_display['authors'] = df_display['authors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )
        
        # Create the data table
        return dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df_display.columns],
            data=df_display.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
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
    
    @app.callback(
        Output("download-tagged-data", "data"),
        [Input("download-data-button", "n_clicks")],
        [State('tagged-data-store', 'data')],
        prevent_initial_call=True,
    )
    def download_tagged_data(n_clicks, df_json):
        """Download the tagged data as CSV."""
        if n_clicks is None or df_json is None:
            raise PreventUpdate
            
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        return dcc.send_data_frame(
            df.to_csv,
            "tagged_papers.csv",
            index=False,
            encoding="utf-8-sig"
        )

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
