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

from .bibtex_processor import process_bibtex_file, BibtexProcessor
from .tag_generator import TagGenerator
from .semantic_search import SemanticSearcher, HybridSemanticSearcher

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
        dcc.Store(id='search-results-store'),
        dcc.Store(id='search-progress-store', data={'progress': 0, 'logs': '', 'status': 'idle'}),
        dcc.Store(id='process-progress-store', data={'progress': 0, 'logs': '', 'status': 'idle'}),
        dcc.Store(id='upload-timestamp', data=None),  # Track upload time to force updates
        dcc.Interval(id='search-interval', interval=500, disabled=True),  # Update every 500ms during search
        dcc.Interval(id='process-interval', interval=500, disabled=True),  # Update every 500ms during processing
        dcc.Download(id="download-wordcloud"),
        dcc.Download(id="download-tagged-data"),
        dcc.Download(id="download-search-results"),
        
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
                # LEFT PANEL: Data Management
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📁 Import & Filter Bibliography"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-bibtex',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select BibTeX or CSV File')
                                ]),
                                accept='.bib,.csv',
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
                            
                            # Bibliography Summary
                            dbc.Card([
                                dbc.CardHeader("📊 Bibliography Summary"),
                                dbc.CardBody([
                                    html.Div(id="bibliography-summary", children="Upload a file to see summary...")
                                ])
                            ], className="mt-3", style={"display": "none"}, id="summary-card"),
                            
                            # Year filtering options
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Min Year", html_for="min-year"),
                                    dbc.Input(
                                        id="min-year",
                                        type="number",
                                        placeholder="e.g., 2020",
                                        className="mb-3"
                                    ),
                                    dbc.FormText("Filter entries from this year onwards"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Max Year", html_for="max-year"),
                                    dbc.Input(
                                        id="max-year",
                                        type="number",
                                        placeholder="e.g., 2023",
                                        className="mb-3"
                                    ),
                                    dbc.FormText("Filter entries up to this year"),
                                ], width=6),
                            ]),
                            
                            html.Hr(),
                            html.P([
                                html.Strong("Apply Filters Only:"), " Prepares your data for semantic search by applying year filters. Use this for quick searching without tag generation."
                            ], className="text-muted small mb-3"),
                            
                            dbc.Button(
                                "📂 Apply Filters Only",
                                id="filter-button",
                                color="secondary",
                                className="w-100 mb-3",
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
                ], md=6),  # Left column - 50% width
                
                # RIGHT PANEL: Analysis & Tag Generation
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="🔍 Interactive Search", tab_id="search-interface-tab", children=[
                            html.Div([
                                # Explanation banner
                                dbc.Alert([
                                    html.P([
                                        "Search your bibliography using natural language queries. This feature works ",
                                        html.Strong("independently of tag generation"), 
                                        " and searches directly through paper titles, abstracts, and any existing metadata."
                                    ], className="mb-0"),
                                ], color="info", className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Search Query", html_for="search-query"),
                                        dbc.Input(
                                            id="search-query",
                                            type="text",
                                            placeholder="e.g., chronic fatigue syndrome, machine learning, climate change",
                                            className="mb-3"
                                        ),
                                        dbc.FormText("Enter any topic or research area - semantic search will find related papers even with different terminology"),
                                    ], width=12),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Search Methods", html_for="search-methods"),
                                        dcc.Checklist(
                                            id="search-methods",
                                            options=[
                                                {"label": "Exact Match (keywords)", "value": "exact"},
                                                {"label": "Fuzzy Match (typos)", "value": "fuzzy"},
                                                {"label": "Semantic Match (AI embeddings)", "value": "semantic"},
                                                {"label": "🚀 Hybrid AI (embeddings + LLM)", "value": "hybrid"},
                                                {"label": "🤖 LLM Only (premium quality)", "value": "llm_only"}
                                            ],
                                            value=["exact", "fuzzy", "semantic"],
                                            inline=False,
                                            className="mb-3"
                                        ),
                                        dbc.FormText([
                                            html.Strong("Exact:"), " Finds papers containing your exact words. ",
                                            html.Strong("Fuzzy:"), " Handles typos and variations. ",
                                            html.Strong("Semantic:"), " Fast AI similarity using embeddings. ",
                                            html.Strong("Hybrid:"), " Best balance - combines embeddings + LLM analysis. ",
                                            html.Strong("LLM Only:"), " Premium quality - GPT analyzes ALL papers (expensive but thorough)."
                                        ]),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("AI Model & Thresholds", html_for="hybrid-model"),
                                        dbc.Select(
                                            id="hybrid-model",
                                            options=[
                                                {"label": "💰 GPT-4o Mini (Recommended)", "value": "gpt-4o-mini"},
                                                {"label": "⚡ GPT-3.5 Turbo", "value": "gpt-3.5-turbo"},
                                                {"label": "🚀 GPT-4o", "value": "gpt-4o"},
                                                {"label": "👑 GPT-4 Turbo (Premium)", "value": "gpt-4-turbo"},
                                            ],
                                            value="gpt-4o-mini",
                                            className="mb-2"
                                        ),
                                        dbc.Label("Semantic Threshold", html_for="semantic-threshold"),
                                        dcc.Slider(
                                            id="semantic-threshold",
                                            min=0.1,
                                            max=1.0,
                                            value=0.7,
                                            step=0.05,
                                            marks={i/10: f"{i/10:.1f}" for i in range(1, 11, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            className="mb-2"
                                        ),
                                        dbc.Label("Fuzzy Threshold", html_for="fuzzy-threshold"),
                                        dcc.Slider(
                                            id="fuzzy-threshold",
                                            min=50,
                                            max=100,
                                            value=80,
                                            step=5,
                                            marks={i: f"{i}" for i in range(50, 101, 25)},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            className="mb-3"
                                        ),
                                    ], width=6),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "🔍 Search Bibliography",
                                            id="search-button",
                                            color="primary",
                                            className="w-100 mb-3",
                                            disabled=True
                                        ),
                                        dbc.FormText(id="search-button-help", children="Upload a bibliography file first to enable search"),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Max Results", html_for="max-results"),
                                        dbc.Input(
                                            id="max-results",
                                            type="number",
                                            min=1,
                                            max=100,
                                            value=20,
                                            className="mb-3"
                                        ),
                                    ], width=3),
                                    dbc.Col([
                                        dbc.Label("LLM Threshold", html_for="llm-threshold"),
                                        dbc.Input(
                                            id="llm-threshold",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=0.5,
                                            value=6.0,
                                            className="mb-3"
                                        ),
                                    ], width=3),
                                ]),
                            ], className="p-3")
                        ]),
                        
                        dbc.Tab(label="🏷️ AI Tags & Word Clouds", tab_id="tagging-interface-tab", children=[
                            html.Div([
                                dbc.Alert([
                                    html.P([
                                        html.Strong("Process & Generate Tags:"), " Applies filters AND generates AI tags for word clouds and tag analysis. Takes longer but enables all visualization features."
                                    ], className="mb-0"),
                                ], color="success", className="mb-3"),
                                
                                # Tag Generation Settings
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
                                    "🏷️ Process & Generate Tags",
                                    id="process-button",
                                    color="primary",
                                    className="w-100 mb-3",
                                    disabled=True
                                ),
                                
                                # Word Cloud Customization (Collapsible)
                                dbc.Collapse([
                                    html.Hr(),
                                    html.H6("Word Cloud Options", className="mb-3"),
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
                                ], id="wordcloud-collapse", is_open=False),
                                
                                dbc.Button(
                                    "⚙️ Show/Hide Word Cloud Options",
                                    id="wordcloud-toggle",
                                    color="secondary",
                                    size="sm",
                                    className="w-100 mt-2"
                                ),
                                
                                # Results section - Word clouds, tag frequencies, and data table
                                html.Hr(className="my-4"),
                                html.H6("📊 Results & Visualizations", className="mb-3"),
                                
                                dbc.Tabs([
                                    dbc.Tab(label="Word Cloud", tab_id="wordcloud-tab", children=[
                                        html.Div(className="text-center mt-3", children=[
                                            dcc.Graph(
                                                id='word-cloud-graph',
                                                config={'displayModeBar': False},
                                                style={
                                                    'width': '100%',
                                                    'height': '500px',
                                                    'border': 'none',
                                                    'borderRadius': '5px',
                                                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                                    'display': 'none'
                                                }
                                            ),
                                            html.Div(id="selected-word", className="h5 mt-3 mb-2"),
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
                                            ], width=8, className="mx-auto"),
                                            className="mb-3"
                                        ),
                                    ]),
                                    
                                    dbc.Tab(label="Tag Frequencies", tab_id="frequencies-tab", children=[
                                        dcc.Graph(id="frequencies-plot", className="mt-3", style={'height': '400px'}),
                                        dbc.Row(
                                            dbc.Col([
                                                dbc.Button(
                                                    "Download Frequencies",
                                                    id="download-freq-button",
                                                    color="success",
                                                    className="mt-3 w-100"
                                                ),
                                            ], width=8, className="mx-auto"),
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
                                            ], width=6),
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
                                            ], width=6),
                                        ]),
                                        html.Div(id="data-table-container", className="mt-3"),
                                    ]),
                                ], id="results-tabs", active_tab="wordcloud-tab"),
                            ], className="p-3")
                        ]),
                        
                    ], id="right-panel-tabs", active_tab="search-interface-tab"),
                ], md=6),  # Right column - 50% width
            ]),
            
            # Search Progress and Results Section (Full Width)
            dbc.Row([
                dbc.Col([
                    # Search Progress Card
                    dbc.Card([
                        dbc.CardHeader("Search Progress"),
                        dbc.CardBody([
                            dbc.Progress(id="search-progress", style={"height": "8px"}, className="mb-2"),
                            html.Div(id="search-progress-text", className="text-center small text-muted mb-2"),
                            dcc.Loading(
                                id="loading-search-logs",
                                type="circle",
                                children=html.Div(
                                    id="search-logs", 
                                    style={
                                        "maxHeight": "200px", 
                                        "overflowY": "auto", 
                                        "whiteSpace": "pre-wrap",
                                        "fontFamily": "monospace",
                                        "fontSize": "0.75rem",
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "10px",
                                        "borderRadius": "5px"
                                    }
                                )
                            )
                        ])
                    ], id="search-progress-card", style={"display": "none"}, className="mb-4"),
                    
                    # Search Results Container
                    html.Div(id="search-results-container", className="mt-3"),
                    
                    # Status and Result Display
                    html.Div(id="search-status"),
                ], width=12)
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

def update_progress(progress_value: float, logger: Optional['DashLogger'] = None, 
                  message: Optional[str] = None) -> Tuple[float, str, List]:
    """Update progress value and log a message.
    
    Args:
        progress_value: Current progress value (0-100)
        logger: Optional logger to record progress
        message: Optional message to log
        
    Returns:
        Tuple with updated progress value, log output, and callback output list
    """
    # Ensure progress is within bounds
    progress = max(0, min(100, progress_value))
    
    # Log the message if logger and message provided
    log_output = ""
    if logger and message:
        log_output = logger.log_info(f"Progress {int(progress)}%: {message}")
    elif logger:
        log_output = logger.log_info(f"Progress {int(progress)}%")
    
    # Prepare the callback output
    callback_output = [
        no_update,  # data-store
        no_update,  # tagged-data-store
        progress,   # process-progress value
        f"Processing: {int(progress)}%",  # process-progress label
        no_update,  # upload-status
        log_output  # processing-logs
    ]
    
    return progress, log_output, callback_output

def create_progress_tracker(total_steps: int = 10) -> callable:
    """Create a progress tracker function that manages progress state.
    
    Args:
        total_steps: Total number of steps in the process
        
    Returns:
        A function that tracks progress through steps
    """
    step_size = 100.0 / total_steps if total_steps > 0 else 10.0
    current_progress = 0.0
    
    def track_progress(step: Optional[int] = None, 
                      increment: bool = False,
                      logger: Optional['DashLogger'] = None,
                      message: Optional[str] = None) -> int:
        """Update progress based on step number or increment.
        
        Args:
            step: Step number (0 to total_steps) or progress percentage (0-100)
            increment: Whether to increment by one step instead of setting absolute progress
            logger: Optional logger to record progress
            message: Optional message to log
            
        Returns:
            Current progress value (0-100)
        """
        nonlocal current_progress
        
        # Calculate new progress
        if step is not None:
            if 0 <= step <= 100:  # Treat as direct percentage
                current_progress = float(step)
            elif 0 <= step <= total_steps:  # Treat as step number
                current_progress = step * step_size
        elif increment:
            current_progress = min(100, current_progress + step_size)
        
        # Update progress and get callback output (unused here)
        current_progress, _, _ = update_progress(current_progress, logger, message)
        
        return int(current_progress)
    
    return track_progress

# Progress tracking functions have been replaced by create_progress_tracker

class DashLogger:
    """Logger class for consistent logging in dashboard callbacks.
    
    Provides methods for tracking processing steps, errors, and progress with
    consistent formatting and behavior.
    """
    
    def __init__(self, max_logs: int = 20):
        """Initialize the logger.
        
        Args:
            max_logs: Maximum number of log entries to retain
        """
        self.logs = []
        self.max_logs = max_logs
    
    def log(self, message: str) -> str:
        """Log a message with timestamp.
        
        Args:
            message: Message to log
            
        Returns:
            Formatted log string with the most recent logs
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        return self._format_logs()
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None) -> str:
        """Log an error message with optional exception details.
        
        Args:
            error_msg: Error message
            exception: Optional exception object
            
        Returns:
            Formatted log string with the error message
        """
        message = f"ERROR: {error_msg}"
        if exception:
            message += f" - {str(exception)}"
        return self.log(message)
    
    def log_success(self, message: str) -> str:
        """Log a success message.
        
        Args:
            message: Success message
            
        Returns:
            Formatted log string with the success message
        """
        return self.log(f"SUCCESS: {message}")
    
    def log_info(self, message: str) -> str:
        """Log an informational message.
        
        Args:
            message: Informational message
            
        Returns:
            Formatted log string with the info message
        """
        return self.log(f"INFO: {message}")
    
    def _format_logs(self) -> str:
        """Format logs for display.
        
        Returns:
            String with the most recent logs joined by newlines
        """
        return "\n".join(self.logs[-self.max_logs:])

def create_logger(max_logs: int = 20) -> DashLogger:
    """Create a logger for tracking processing steps.
    
    Args:
        max_logs: Maximum number of log entries to retain
        
    Returns:
        A logger object
    """
    return DashLogger(max_logs)

class SearchProgressTracker:
    """Thread-safe progress tracker for search operations."""
    
    def __init__(self):
        """Initialize the progress tracker."""
        import threading
        self.lock = threading.Lock()
        self.progress = 0
        self.logs = []
        self.status = 'idle'
        self.max_logs = 30
        
    def update(self, progress: int, log_message: str = None, status: str = None):
        """Update progress with thread safety."""
        with self.lock:
            self.progress = progress
            if log_message:
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.logs.append(f"[{timestamp}] {log_message}")
                # Keep only recent logs
                if len(self.logs) > self.max_logs:
                    self.logs = self.logs[-self.max_logs:]
            if status:
                self.status = status
                
    def get_state(self):
        """Get current state with thread safety."""
        with self.lock:
            return {
                'progress': self.progress,
                'logs': '\n'.join(self.logs),
                'status': self.status
            }

# Global progress trackers
search_progress = SearchProgressTracker()
process_progress = SearchProgressTracker()

class ProgressLogger(DashLogger):
    """Enhanced logger that updates the progress tracker."""
    
    def __init__(self, progress_tracker: SearchProgressTracker, max_logs: int = 20):
        """Initialize with a progress tracker."""
        super().__init__(max_logs)
        self.progress_tracker = progress_tracker
        self.current_progress = 0
        
    def log_info(self, message: str) -> str:
        """Log info and update progress tracker."""
        result = super().log_info(message)
        self.progress_tracker.update(self.current_progress, message)
        return result
        
    def log_success(self, message: str) -> str:
        """Log success and update progress tracker."""
        result = super().log_success(message)
        self.progress_tracker.update(self.current_progress, message)
        return result
        
    def log_error(self, message: str, error: Exception = None) -> str:
        """Log error and update progress tracker."""
        result = super().log_error(message, error)
        self.progress_tracker.update(self.current_progress, message, status='error')
        return result
        
    def set_progress(self, progress: int):
        """Set the current progress value."""
        self.current_progress = progress
        self.progress_tracker.update(progress)

def parse_tag_string(tag_string: str) -> List[str]:
    """Parse a comma-separated tag string into a list of cleaned tags.
    
    Args:
        tag_string: Comma-separated string of tags
        
    Returns:
        List of clean tag strings with empty and 'nan' values removed
    """
    if not isinstance(tag_string, str):
        return []
        
    # Split by comma, strip whitespace, and filter out empty or 'nan' tags
    tags = []
    for tag in tag_string.split(','):
        tag = tag.strip()
        if tag and tag.lower() != 'nan':
            tags.append(tag)
            
    return tags

def get_tags_from_series(tag_series: pd.Series) -> List[str]:
    """Extract all tags from a pandas Series of tag strings.
    
    Args:
        tag_series: Pandas Series containing tag strings
        
    Returns:
        List of all tags (may contain duplicates)
    """
    all_tags = []
    
    for tag_string in tag_series.dropna():
        all_tags.extend(parse_tag_string(tag_string))
        
    return all_tags

def count_tag_frequencies(df: pd.DataFrame) -> Dict[str, int]:
    """Count the frequency of each tag in the DataFrame.
    
    Args:
        df: DataFrame containing a 'tags' column
        
    Returns:
        Dictionary mapping tags to their frequencies
    """
    if 'tags' not in df.columns:
        return {}
        
    # Get all tags from the DataFrame
    all_tags = get_tags_from_series(df['tags'])
    
    # Count occurrences
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
    return tag_counts

def extract_unique_tags(df: pd.DataFrame) -> set:
    """Extract unique tags from the DataFrame.
    
    Args:
        df: DataFrame containing a 'tags' column
        
    Returns:
        Set of unique tags
    """
    if 'tags' not in df.columns:
        return set()
        
    # Get all tags and convert to a set to remove duplicates
    return set(get_tags_from_series(df['tags']))

def create_paper_cards(papers: List[Dict], show_tags: bool = False, 
                     show_journal: bool = True, max_title_length: int = 100) -> List[dbc.Card]:
    """Create Dash cards for papers.
    
    Args:
        papers: List of paper data dictionaries
        show_tags: Whether to display tags in the card
        show_journal: Whether to display journal information
        max_title_length: Maximum length for paper titles before truncating
        
    Returns:
        List of Dash Card components
    """
    paper_cards = []
    
    for paper in papers:
        # Prepare title (truncate if too long)
        title = paper.get('title', 'No title')
        if len(title) > max_title_length:
            title = title[:max_title_length] + '...'
            
        # Prepare journal info if available and requested
        journal_info = None
        if show_journal and paper.get('journal'):
            journal_text = paper['journal']
            if len(journal_text) > 50:  # Truncate long journal names
                journal_text = journal_text[:47] + '...'
            journal_info = html.P(journal_text, className="text-muted font-italic")
            
        # Prepare tag badges if available and requested
        tag_elements = None
        if show_tags and paper.get('tags'):
            tags = parse_tag_string(paper['tags'])
            if tags:
                tag_badges = [
                    dbc.Badge(tag, color="info", className="mr-1 mb-1")
                    for tag in tags[:5]  # Limit to first 5 tags
                ]
                if len(tags) > 5:
                    tag_badges.append(dbc.Badge(f"+{len(tags)-5} more", color="secondary"))
                tag_elements = html.Div(tag_badges, className="mt-2")
        
        # Prepare publication ID if available
        pub_id = get_field_case_insensitive(pd.Series(paper), 'publication_id')
        pub_id_info = None
        if pub_id:
            pub_id_info = html.P([
                html.Small([
                    html.Strong("ID: "), 
                    pub_id
                ])
            ], className="text-muted mb-1")
        
        # Create card with all elements
        card_elements = [
            dbc.CardHeader([
                html.Span(f"{paper.get('year', 'N/A')}", className="float-right"),
                html.Strong(paper.get('authors', 'Unknown'))
            ]),
            dbc.CardBody([
                html.H5(title, className="card-title"),
                pub_id_info,
                journal_info,
                tag_elements,
                html.Div([
                    dbc.Button("View Paper", href=paper.get('url', '#'), color="primary", 
                             external_link=True, target="_blank",
                             className="mt-2") 
                    if paper.get('url') and paper.get('url') != '#' else None
                ], className="text-right")
            ])
        ]
        
        # Filter out None elements
        card_elements = [elem for elem in card_elements if elem is not None]
        
        card = dbc.Card(card_elements, className="mb-3")
        paper_cards.append(card)
        
    return paper_cards

def df_from_json_store(df_json: Optional[str], logger: Optional['DashLogger'] = None, 
                        required_columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Convert stored JSON data to DataFrame with validation and error handling.
    
    Args:
        df_json: JSON string representing DataFrame
        logger: Optional logger for recording conversion issues
        required_columns: Optional list of column names that must be present
        
    Returns:
        DataFrame or None if conversion fails
    """
    if not df_json:
        if logger:
            logger.log_error("Empty JSON data provided")
        return None
        
    try:
        # Convert JSON to DataFrame
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        # Check if the DataFrame is empty
        if df.empty:
            if logger:
                logger.log_error("Converted DataFrame is empty")
            return None
            
        # Validate required columns if specified
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                if logger:
                    logger.log_error(f"Missing required columns: {', '.join(missing_columns)}")
                return None
                
        # Log success if logger provided
        if logger:
            logger.log_info(f"Successfully converted JSON to DataFrame with {len(df)} rows and {len(df.columns)} columns")
            
        return df
        
    except ValueError as e:
        if logger:
            logger.log_error("JSON parsing error", e)
        return None
    except Exception as e:
        if logger:
            logger.log_error("Unexpected error during DataFrame conversion", e)
        return None

def create_paper_info(row: pd.Series, include_abstract: bool = False) -> Dict[str, str]:
    """Create a standardized paper information dictionary from a DataFrame row.
    
    Args:
        row: A pandas Series (DataFrame row) containing paper data
        include_abstract: Whether to include the abstract in the result
        
    Returns:
        Dictionary with standardized paper information
    """
    # Extract author information with proper handling
    author_text = row.get('author', 'Unknown')
    if pd.isna(author_text):
        author_text = 'Unknown'
        
    # Handle multiple authors (get first author or full list)
    if ' and ' in author_text:
        first_author = author_text.split(' and ')[0]
        authors_full = author_text
    else:
        first_author = author_text
        authors_full = author_text
    
    # Create basic paper info dictionary
    paper_info = {
        'title': row.get('title', 'No title') if pd.notna(row.get('title')) else 'No title',
        'year': row.get('year', 'N/A') if pd.notna(row.get('year')) else 'N/A',
        'authors': first_author,
        'authors_full': authors_full,
        'journal': row.get('journal', '') if pd.notna(row.get('journal')) else '',
        'url': row.get('url', '#') if pd.notna(row.get('url')) else '#',
        'entry_type': row.get('ENTRYTYPE', '') if pd.notna(row.get('ENTRYTYPE')) else ''
    }
    
    # Add publication_id if available (case-insensitive)
    pub_id = get_field_case_insensitive(row, 'publication_id')
    if pub_id:
        paper_info['publication_id'] = pub_id
    
    # Add abstract if requested
    if include_abstract and 'abstract' in row and pd.notna(row['abstract']):
        paper_info['abstract'] = row['abstract']
    
    # Add tags if available
    if 'tags' in row and pd.notna(row['tags']):
        paper_info['tags'] = row['tags']
    
    return paper_info

def get_field_case_insensitive(row: pd.Series, field_name: str) -> str:
    """Get a field value from a row with case-insensitive matching.
    
    Args:
        row: Pandas Series (DataFrame row)
        field_name: Field name to search for (case insensitive)
        
    Returns:
        Field value if found, empty string otherwise
    """
    if not isinstance(row, pd.Series):
        return ""
    
    # Convert to dict for easier searching
    row_dict = row.to_dict()
    
    # First try exact match
    if field_name in row_dict:
        value = row_dict[field_name]
        return str(value) if value is not None and pd.notna(value) else ""
    
    # Try case-insensitive match
    field_lower = field_name.lower()
    for key, value in row_dict.items():
        if str(key).lower() == field_lower:
            return str(value) if value is not None and pd.notna(value) else ""
    
    return ""

def find_papers_with_tag(df: pd.DataFrame, tag: str) -> List[Dict]:
    """Find papers containing a specific tag.
    
    Args:
        df: DataFrame with papers data
        tag: Tag to search for
        
    Returns:
        List of dictionaries with paper information
    """
    if 'tags' not in df.columns:
        return []
    
    papers = []
    for idx, row in df.iterrows():
        if pd.notna(row.get('tags')):
            # Use our new tag parsing utility
            tags = parse_tag_string(row['tags'])
            if tag in tags:
                papers.append(create_paper_info(row))
    
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
            False,
            dbc.Alert(f"File '{filename}' uploaded successfully!", color="success")
        )
    
    return "", True, True, ""

def handle_process_bibtex(n_clicks: Optional[int], filename: Optional[str], 
                         tag_sample_size: int, max_entries_to_tag: int,
                         model: str, upload_dir: Path) -> Tuple[Union[str, Any], Union[str, Any], 
                                                             int, str, Optional[dbc.Alert], str]:
    """Process BibTeX entries from a file and generate tags.
    
    Args:
        n_clicks: Number of button clicks
        filename: Name of the uploaded file
        tag_sample_size: Number of samples to take for tag generation
        max_entries_to_tag: Maximum number of entries to tag
        model: Model to use for tag generation
        upload_dir: Directory containing uploads
        
    Returns:
        Tuple containing data store JSON, processed status, progress, status text, alert component, and log text
    """
    # Prevent update if no input or button not clicked
    if not filename or n_clicks is None or n_clicks == 0:
        raise PreventUpdate
        
    # Create progress tracker and logger
    progress_callback = create_progress_tracker()
    logger = create_logger()
    
    # Initialize logging
    logs = logger.log_info(f"Starting to process {filename} file")
    
    try:
        # Update progress
        progress = progress_callback(10)
        
        # Create file path
        file_path = upload_dir / filename
        
        # Validate file exists
        validation_error = validate_file_exists(file_path, logger)
        if validation_error:
            return validation_error
            
        # Update progress
        progress = progress_callback(20)
        
        # Process BibTeX entries
        df, error_msg = process_bibtex_entries(file_path, logger)
        if error_msg:
            logs = logger.log_error(error_msg)
            return (
                no_update,
                no_update,
                0,
                error_msg,
                dbc.Alert(error_msg, color="danger"),
                logs
            )
            
        # Update progress
        progress = progress_callback(40)
        
        # Generate and assign tags
        df, percentage = generate_and_assign_tags(
            df, model, tag_sample_size, max_entries_to_tag, logger, filename)
            
        # Update progress
        progress = progress_callback(80)
        
        # Convert to JSON for storage
        df_json = df.to_json(date_format='iso', orient='split')
        progress = progress_callback(90)
        
        # Calculate percentage of tagged entries
        tagged_count = df['tags'].notna().sum()
        total_count = len(df)
        
        # Create success message
        status_msg = f"Successfully processed {total_count} entries, {tagged_count} with tags ({percentage}%)."
        logs = logger.log_success(status_msg)
        alert = dbc.Alert(status_msg, color="success")
        
        # Complete progress
        progress = progress_callback(100)
        
        return df_json, "done", progress, status_msg, alert, logs
    except Exception as e:
        # Handle unexpected errors
        error_msg = "Unexpected error during processing"
        progress = progress_callback(0)
        logs = logger.log_error(error_msg, e)
        alert = dbc.Alert(f"{error_msg}: {str(e)}", color="danger")
        
        return no_update, no_update, progress, error_msg, alert, logs

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
    # Handle callback context and defaults
    result = handle_wordcloud_context(n_clicks, initial_load_ts, df_json, max_words, color_scheme, bg_color, wc_type)
    if result:
        return result
    
    # If no data, show upload message
    if df_json is None:
        return go.Figure(), None, True, "Please upload and process a BibTeX file first.", True, {'display': 'none'}
    
    try:
        # Get the data
        df = df_from_json_store(df_json)
        if df is None:
            return go.Figure(), None, True, "Error loading data.", True, {'display': 'block'}
        
        # Prepare tags and paper mappings for word cloud
        tag_counts_limited, word_to_papers, error_msg = prepare_tags_for_word_cloud(df, max_words)
        
        if error_msg:
            return go.Figure(), None, True, error_msg, True, {'display': 'block'}
            
        # Generate word cloud image
        img_str = create_wordcloud_image(tag_counts_limited, bg_color, color_scheme)
        
        if not img_str:
            return go.Figure(), None, True, "Error generating word cloud image.", True, {'display': 'block'}
        
        # Generate appropriate visualization based on type
        if wc_type == 'static':
            return generate_static_word_cloud(img_str, bg_color)
        else:  # Interactive mode
            return generate_interactive_word_cloud(tag_counts_limited, word_to_papers, bg_color, img_str)
                
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
    columns_to_show = ['year', 'title', 'authors', 'journal', 'publication_id', 'doi']
    available_columns = []
    
    # Check for columns with case-insensitive matching
    for desired_col in columns_to_show:
        # First try exact match
        if desired_col in df.columns:
            available_columns.append(desired_col)
        else:
            # Try case-insensitive match for publication_id
            if desired_col.lower() == 'publication_id':
                for col in df.columns:
                    if col.lower() == 'publication_id':
                        # Add the column with its original case but rename it for display
                        df['publication_id'] = df[col]
                        available_columns.append('publication_id')
                        break
            else:
                # For other columns, check if they exist exactly
                if desired_col in df.columns:
                    available_columns.append(desired_col)
    
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

def validate_file_exists(file_path: Path, logger: 'DashLogger') -> Optional[Tuple]:
    """Validate that a file exists and return error if not.
    
    Args:
        file_path: Path to the file
        logger: Logger object for tracking
        
    Returns:
        Error tuple if file not found, None if file exists
    """
    if not file_path.exists():
        error_msg = "File not found"
        return (
            no_update,
            no_update,
            0,
            error_msg,
            dbc.Alert(error_msg, color="danger"),
            logger.log_error(error_msg)
        )
    return None

def process_bibtex_entries(file_path: Path, logger: 'DashLogger') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Process bibliography entries from a BibTeX or CSV file.
    
    Args:
        file_path: Path to the BibTeX or CSV file
        logger: Logger for tracking progress
        
    Returns:
        DataFrame of bibliography entries and any error message
    """
    try:
        # Detect file format and read accordingly
        file_ext = file_path.suffix.lower()
        logger.log_info(f"Reading {file_ext.upper().replace('.', '')} file: {file_path.name}")
        
        if file_ext == '.csv':
            # Process CSV file
            logger.log_info("Parsing CSV entries...")
            try:
                df = pd.read_csv(file_path)
                entries = df.to_dict('records')
                
                # Normalize entries to match expected format
                normalized_entries = []
                for i, entry in enumerate(entries):
                    clean_entry = {}
                    for key, value in entry.items():
                        if pd.notna(value):
                            clean_entry[key.lower()] = str(value)
                        else:
                            clean_entry[key.lower()] = ''
                    
                    # Ensure ID field exists
                    if 'id' not in clean_entry and 'ID' not in clean_entry:
                        clean_entry['ID'] = f"entry_{i + 1}"
                    elif 'id' in clean_entry:
                        clean_entry['ID'] = clean_entry['id']
                    
                    normalized_entries.append(clean_entry)
                
                entries = normalized_entries
                logger.log_info(f"Found {len(entries)} entries")
            except Exception as e:
                error_msg = "Failed to parse CSV"
                logger.log_error(error_msg, e)
                return None, error_msg
        else:
            # Process BibTeX file (default)
            with open(file_path, 'r', encoding='utf-8') as f:
                bibtex_str = f.read()
                
            # Parse BibTeX data
            logger.log_info("Parsing BibTeX entries...")
            try:
                bib_database = bibtexparser.loads(bibtex_str)
                entries = bib_database.entries
                logger.log_info(f"Found {len(entries)} entries")
            except Exception as e:
                error_msg = "Failed to parse BibTeX"
                logger.log_error(error_msg, e)
                return None, error_msg
            
        if not entries:
            error_msg = f"No entries found in the {file_ext.upper().replace('.', '')} file"
            logger.log_error(error_msg)
            return None, error_msg
            
        # Convert to DataFrame
        logger.log_info("Converting entries to DataFrame...")
        df = pd.DataFrame(entries)
        
        # Filter out entries without abstracts if needed
        if 'abstract' in df.columns:
            has_abstract = df['abstract'].notna() & (df['abstract'] != '')
            abstract_count = has_abstract.sum()
            logger.log_info(f"Found {abstract_count} entries with abstracts")
        
        logger.log_success(f"Successfully processed {len(df)} entries")
        return df, None
        
    except Exception as e:
        error_msg = "Failed to process BibTeX file"
        logger.log_error(error_msg, e)
        return None, error_msg

def generate_and_assign_tags(df: pd.DataFrame, model: str, tag_sample_size: int, 
                           max_entries_to_tag: int, logger: 'DashLogger', 
                           file_name: str) -> Tuple[pd.DataFrame, int]:
    """Generate and assign tags to DataFrame entries based on abstracts.
    
    Args:
        df: DataFrame of BibTeX entries
        model: Model to use for tag generation
        tag_sample_size: Number of samples to take for tag generation
        max_entries_to_tag: Maximum number of entries to tag
        logger: Logger object for tracking progress
        file_name: Name of the BibTeX file
        
    Returns:
        Tuple with tagged DataFrame and percentage of tagged entries
    """
    # Ensure model is valid
    if model != 'gpt-3.5-turbo' and model != 'gpt-4':
        model = 'gpt-3.5-turbo'
        logger.log_info("Warning: Invalid model specified, using gpt-3.5-turbo")
        
    # Filter entries with abstracts
    try:
        logger.log_info("Filtering entries with abstracts...")
        
        # Create columns if not present
        if 'tags' not in df.columns:
            df['tags'] = np.nan
            
        # Count entries before filtering
        total_entries = len(df)
        
        # Filter entries with abstracts
        if 'abstract' in df.columns:
            df_with_abstracts = df[df['abstract'].notna() & (df['abstract'] != '')].copy()
            logger.log_info(f"Found {len(df_with_abstracts)} entries with abstracts")
        else:
            df_with_abstracts = pd.DataFrame()
            logger.log_info("No abstracts found in the data")
            
        # Skip tag generation if no entries have abstracts
        if df_with_abstracts.empty:
            logger.log_info("Skipping tag generation: no entries with abstracts")
            return df, 0
            
        # Limit entries to tag
        if max_entries_to_tag > 0 and len(df_with_abstracts) > max_entries_to_tag:
            logger.log_info(f"Limiting tag generation to first {max_entries_to_tag} entries")
            df_to_tag = df_with_abstracts.head(max_entries_to_tag).copy()
        else:
            df_to_tag = df_with_abstracts.copy()
            
        # Generate tags for each entry with abstract
        logger.log_info(f"Generating tags for {len(df_to_tag)} entries using {model}...")
        
        # Sample entries to generate tags from
        if tag_sample_size > 0 and len(df_to_tag) > tag_sample_size:
            logger.log_info(f"Sampling {tag_sample_size} entries for tag generation")
            df_sample = df_to_tag.sample(n=tag_sample_size)
        else:
            df_sample = df_to_tag
            
        # Generate tags from sample
        sample_tags = []
        for _, row in df_sample.iterrows():
            try:
                record_type = row.get('ENTRYTYPE', '')
                title = row.get('title', 'Untitled')
                abstract = row.get('abstract', '')
                year = row.get('year', '')
                
                # Generate tags for this entry
                entry_tags = generate_tags(record_type, title, abstract, year, model)
                
                # Combine all tags
                if entry_tags:
                    sample_tags.extend(entry_tags)
            except Exception as e:
                logger.log_error(f"Error generating tags for entry", e)
                continue
                
        # Remove duplicates and sort
        all_tags = sorted(list(set(sample_tags)))
        logger.log_info(f"Generated {len(all_tags)} unique tags from sample")
        
        # Use tags to classify all entries
        logger.log_info("Classifying all entries with the generated tags...")
        for idx, row in df_to_tag.iterrows():
            try:
                record_type = row.get('ENTRYTYPE', '')
                title = row.get('title', 'Untitled')
                abstract = row.get('abstract', '')
                year = row.get('year', '')
                
                # Assign tags to this entry
                entry_tags = assign_tags(record_type, title, abstract, year, all_tags, model)
                
                # Store tags in DataFrame
                if entry_tags:
                    df.at[idx, 'tags'] = ', '.join(entry_tags)
            except Exception as e:
                logger.log_error(f"Error assigning tags for entry", e)
                continue
                
        # Count entries that have tags
        tagged_entries = df['tags'].notna().sum()
        percent_tagged = round((tagged_entries / total_entries) * 100)
        logger.log_success(f"Tagged {tagged_entries} out of {total_entries} entries ({percent_tagged}%)")
        
        return df, percent_tagged
    except Exception as e:
        logger.log_error("Error in tag generation", e)
        return df, 0

def prepare_tags_for_word_cloud(df: pd.DataFrame, max_words: int) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, List[Dict]]], Optional[str]]:
    """Extract and prepare tags for word cloud generation.
    
    Args:
        df: DataFrame containing tag data
        max_words: Maximum number of words to include
        
    Returns:
        Tuple containing tag frequencies dictionary, word-to-papers mapping, and error message if any
    """
    if 'tags' not in df.columns or df['tags'].isna().all():
        return None, None, "No tags found in the data."
    
    # Get tag frequencies
    tag_counts = count_tag_frequencies(df)
    
    if not tag_counts:
        return None, None, "No valid tags found."
    
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
        return None, None, "No papers found for tags."
    
    return tag_counts_limited, word_to_papers, None

def create_wordcloud_image(tag_counts_limited: Dict[str, int], bg_color: str, color_scheme: str) -> Optional[str]:
    """Create a word cloud image from tag counts.
    
    Args:
        tag_counts_limited: Dictionary of tags and their frequencies
        bg_color: Background color for the word cloud
        color_scheme: Color scheme for the word cloud
        
    Returns:
        Base64 encoded image string or None if error
    """
    try:
        # Create a word cloud with specified parameters
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color=bg_color,
            max_words=len(tag_counts_limited),
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
        
        return img_str
    except Exception as e:
        import traceback
        print(f"Error generating word cloud image: {str(e)}")
        print(traceback.format_exc())
        return None

def handle_wordcloud_context(n_clicks: Optional[int], initial_load_ts: Optional[int], 
                        df_json: Optional[str], max_words: int, color_scheme: str, 
                        bg_color: str, wc_type: str) -> Optional[Tuple]:
    """Handle callback context for word cloud generation.
    
    Args:
        n_clicks: Number of button clicks
        initial_load_ts: Initial load timestamp
        df_json: JSON string containing DataFrame
        max_words: Maximum number of words in cloud
        color_scheme: Color scheme for word cloud
        bg_color: Background color
        wc_type: Type of word cloud (static or interactive)
        
    Returns:
        Tuple containing figure etc. if context is handled, None if normal processing should continue
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
        
    return None

def register_callbacks(app: dash.Dash, upload_dir: Path) -> None:
    """Register all Dash callbacks.
    
    Args:
        app: Dash application
        upload_dir: Directory for uploaded files
    """
    @app.callback(
        [Output('upload-filename', 'children'),
         Output('process-button', 'disabled'),
         Output('filter-button', 'disabled'),
         Output('upload-status', 'children'),
         Output('upload-timestamp', 'data')],
        [Input('upload-bibtex', 'contents')],
        [State('upload-bibtex', 'filename')]
    )
    def update_upload_status(contents, filename):
        """Update the upload status and enable/disable the process button."""
        result = handle_upload_status(contents, filename, upload_dir)
        # Add timestamp to force update of dependent callbacks
        timestamp = datetime.now().isoformat() if contents else None
        return result + (timestamp,)
    
    @app.callback(
        [Output('data-store', 'data'),
         Output('tagged-data-store', 'data'),
         Output('process-progress', 'value'),
         Output('process-progress', 'label'),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('processing-logs', 'children'),
         Output('process-interval', 'disabled', allow_duplicate=True)],
        [Input('process-button', 'n_clicks')],
        [State('upload-bibtex', 'filename'),
         State('tag-sample-size', 'value'),
         State('max-entries-to-tag', 'value'),
         State('model-select', 'value'),
         State('min-year', 'value'),
         State('max-year', 'value')],
        prevent_initial_call=True
    )
    def process_bibtex(n_clicks, filename, tag_sample_size, max_entries_to_tag, model, min_year, max_year):
        """Process the uploaded BibTeX file and generate tags with detailed logging."""
        if n_clicks is None or not filename:
            raise PreventUpdate
            
        # Reset and initialize progress tracker
        process_progress.__init__()
        process_progress.update(0, "Starting tag generation...", status='processing')
        
        file_path = upload_dir / filename
        logger = ProgressLogger(process_progress)
        
        if not file_path.exists():
            error_msg = "Error: File not found"
            logs = logger.log_error(error_msg)
            process_progress.update(0, status='error')
            return (
                no_update,
                no_update,
                0,
                error_msg,
                dbc.Alert(error_msg, color="danger"),
                logs,
                True  # Disable interval
            )
        
        try:
            # Process the BibTeX file
            try:
                logs = logger.log_info(f"Starting processing of {filename}")
                logger.set_progress(10)
                entries = process_bibtex_file(file_path)
                logger.set_progress(20)
                logs = logger.log_info(f"Found {len(entries)} entries in the file")
                
                # Apply year filtering if specified
                if min_year or max_year:
                    processor = BibtexProcessor()
                    processor.entries = entries
                    
                    filters = {}
                    if min_year:
                        filters['min_year'] = min_year
                        log = update_log(f"Filtering entries from year {min_year} onwards")
                    if max_year:
                        filters['max_year'] = max_year
                        log = update_log(f"Filtering entries up to year {max_year}")
                    
                    original_count = len(entries)
                    entries = processor.filter_entries(**filters)
                    filtered_count = len(entries)
                    
                    if filtered_count < original_count:
                        log = update_log(f"Year filtering: {original_count} entries -> {filtered_count} entries")
                
                if not entries:
                    error_msg = "Error: No entries found after filtering"
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
    
    # Bibliography summary callback
    @app.callback(
        [Output('bibliography-summary', 'children'),
         Output('summary-card', 'style')],
        [Input('upload-timestamp', 'data')],  # Trigger when upload timestamp changes
        [State('upload-bibtex', 'filename')],
        prevent_initial_call=True
    )
    def update_bibliography_summary(upload_timestamp, filename):
        """Generate and display bibliography summary after file upload."""
        # Check if upload was successful
        if not upload_timestamp or not filename:
            return "Upload a file to see summary...", {"display": "none"}
        
        try:
            # Process the uploaded file to get basic info
            file_path = upload_dir / filename
            
            # Add a small delay to ensure file is written
            import time
            time.sleep(0.1)
            
            if not file_path.exists():
                # Try to list what files are in the directory
                files_in_dir = list(upload_dir.glob("*"))
                error_msg = f"File '{filename}' not found in uploads. Files in directory: {[f.name for f in files_in_dir]}"
                return dbc.Alert(error_msg, color="warning"), {"display": "block"}
            
            # Process the file
            entries = process_bibtex_file(file_path)
            df = pd.DataFrame(entries)
            
            # Calculate statistics
            total_entries = len(df)
            
            # Count entries with MEANINGFUL content (not just non-null)
            has_title = 0
            has_abstract = 0
            has_author = 0
            has_year = 0
            has_journal = 0
            has_doi = 0
            
            if 'title' in df.columns:
                has_title = df['title'].notna().sum()
                # Filter out empty or very short titles
                meaningful_titles = df['title'].dropna().str.strip().str.len() > 10
                has_title = meaningful_titles.sum()
            
            if 'abstract' in df.columns:
                # Filter out empty, very short, or placeholder abstracts
                meaningful_abstracts = (
                    df['abstract'].notna() & 
                    (df['abstract'].str.strip() != '') &
                    (df['abstract'].str.len() > 50) &  # At least 50 characters
                    (~df['abstract'].str.lower().str.contains('no abstract|abstract not available|n/a', na=False))
                )
                has_abstract = meaningful_abstracts.sum()
            
            if 'author' in df.columns:
                meaningful_authors = (
                    df['author'].notna() & 
                    (df['author'].str.strip() != '') &
                    (df['author'].str.len() > 2)
                )
                has_author = meaningful_authors.sum()
            
            if 'year' in df.columns:
                # Only count valid 4-digit years between 1000-2100
                years_numeric = pd.to_numeric(df['year'], errors='coerce')
                valid_years = (years_numeric >= 1000) & (years_numeric <= 2100)
                has_year = valid_years.sum()
            
            if 'journal' in df.columns:
                meaningful_journals = (
                    df['journal'].notna() & 
                    (df['journal'].str.strip() != '') &
                    (df['journal'].str.len() > 2)
                )
                has_journal = meaningful_journals.sum()
            
            if 'doi' in df.columns:
                # DOI should look like a DOI (contain at least one dot and slash)
                meaningful_dois = (
                    df['doi'].notna() & 
                    (df['doi'].str.strip() != '') &
                    df['doi'].str.contains(r'10\.', na=False)  # DOIs start with "10."
                )
                has_doi = meaningful_dois.sum()
            
            # Year range (only from valid years)
            year_range = "N/A"
            if 'year' in df.columns and has_year > 0:
                years = pd.to_numeric(df['year'], errors='coerce')
                valid_years = years[(years >= 1000) & (years <= 2100)].dropna()
                if len(valid_years) > 0:
                    min_year = int(valid_years.min())
                    max_year = int(valid_years.max())
                    year_range = f"{min_year} - {max_year}"
            
            # Create summary display
            summary_content = [
                dbc.Row([
                    dbc.Col([
                        html.H6("📄 Total Entries", className="text-primary mb-1"),
                        html.H4(f"{total_entries:,}", className="mb-0")
                    ], width=4),
                    dbc.Col([
                        html.H6("📅 Year Range", className="text-primary mb-1"),
                        html.H4(year_range, className="mb-0")
                    ], width=4),
                    dbc.Col([
                        html.H6("📝 With Abstracts", className="text-primary mb-1"),
                        html.H4(f"{has_abstract:,} ({has_abstract/total_entries*100:.1f}%)", className="mb-0")
                    ], width=4),
                ], className="mb-3"),
                
                html.Hr(),
                
                html.H6("Field Completeness:", className="mb-2"),
                dbc.Progress([
                    dbc.Progress(value=has_title/total_entries*100, label=f"Titles: {has_title:,}", bar=True, color="success"),
                ], className="mb-2"),
                dbc.Progress([
                    dbc.Progress(value=has_author/total_entries*100, label=f"Authors: {has_author:,}", bar=True, color="info"),
                ], className="mb-2"),
                dbc.Progress([
                    dbc.Progress(value=has_year/total_entries*100, label=f"Years: {has_year:,}", bar=True, color="warning"),
                ], className="mb-2"),
                dbc.Progress([
                    dbc.Progress(value=has_journal/total_entries*100, label=f"Journals: {has_journal:,}", bar=True, color="primary"),
                ], className="mb-2"),
                dbc.Progress([
                    dbc.Progress(value=has_doi/total_entries*100, label=f"DOIs: {has_doi:,}", bar=True, color="secondary"),
                ], className="mb-0"),
                
                html.Hr(),
                
                # Add data quality insights
                html.H6("🔍 Search Readiness:", className="mb-2"),
                html.Div([
                    dbc.Badge(f"{has_abstract:,} papers ready for semantic search", 
                             color="success" if has_abstract > total_entries * 0.5 else "warning", 
                             className="me-2 mb-1"),
                    dbc.Badge(f"{has_year:,} with valid years for filtering", 
                             color="info", className="me-2 mb-1"),
                    dbc.Badge(f"{total_entries - has_abstract:,} missing meaningful abstracts", 
                             color="secondary", className="me-2 mb-1"),
                ]),
                
                html.Hr(),
                html.Small([
                    html.Strong("💡 Quality Check: "), 
                    f"Only papers with meaningful abstracts (50+ chars) can be effectively searched. " +
                    f"You have {has_abstract:,} papers ({has_abstract/total_entries*100:.1f}%) ready for semantic search. " +
                    (f"Consider using Publication Enricher to add missing abstracts." if has_abstract < total_entries * 0.8 else "Great abstract coverage!")
                ], className="text-muted")
            ]
            
            return summary_content, {"display": "block"}
            
        except Exception as e:
            error_content = [
                dbc.Alert(f"Error analyzing file: {str(e)}", color="warning", className="mb-0")
            ]
            return error_content, {"display": "block"}
    
    # Filter-only callback for applying filters without tag generation
    @app.callback(
        [Output('data-store', 'data', allow_duplicate=True),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('processing-logs', 'children', allow_duplicate=True),
         Output('process-interval', 'disabled', allow_duplicate=True),
         Output('process-progress', 'value', allow_duplicate=True),
         Output('process-progress-text', 'children', allow_duplicate=True)],
        [Input('filter-button', 'n_clicks')],
        [State('upload-bibtex', 'filename'),
         State('min-year', 'value'),
         State('max-year', 'value')],
        prevent_initial_call=True
    )
    def apply_filters_only(n_clicks, filename, min_year, max_year):
        """Apply filters to the uploaded file without generating tags."""
        if n_clicks is None or not filename:
            raise PreventUpdate
            
        # Reset and initialize progress tracker
        process_progress.__init__()
        process_progress.update(0, "Starting file processing...", status='processing')
        
        file_path = upload_dir / filename
        logger = ProgressLogger(process_progress)
        
        # Enable interval for real-time updates (will be returned in first callback response)
        enable_interval = False
        
        logs = logger.log_info(f"Starting to filter {filename}")
        logger.set_progress(10)
        
        if not file_path.exists():
            error_msg = "Error: File not found"
            logs = logger.log_error(error_msg)
            process_progress.update(0, status='error')
            return (
                no_update,
                dbc.Alert(error_msg, color="danger"),
                logs,
                True,  # Disable interval
                0,
                "Error"
            )
        
        try:
            # Process the file
            logs = logger.log_info(f"Reading {filename}...")
            logger.set_progress(20)
            entries = process_bibtex_file(file_path)
            logger.set_progress(40)
            logs = logger.log_info(f"Found {len(entries)} entries in the file")
            
            # Apply year filtering if specified
            if min_year or max_year:
                logger.set_progress(50)
                processor = BibtexProcessor()
                processor.entries = entries
                
                filters = {}
                if min_year:
                    filters['min_year'] = min_year
                    logs = logger.log_info(f"Filtering entries from year {min_year} onwards")
                if max_year:
                    filters['max_year'] = max_year
                    logs = logger.log_info(f"Filtering entries up to year {max_year}")
                
                logger.set_progress(60)
                original_count = len(entries)
                entries = processor.filter_entries(**filters)
                filtered_count = len(entries)
                
                if filtered_count < original_count:
                    logs = logger.log_info(f"Year filtering: {original_count} entries -> {filtered_count} entries")
                else:
                    logs = logger.log_info("No entries were filtered out")
            
            if not entries:
                error_msg = "Error: No entries found after filtering"
                logs = logger.log_error(error_msg)
                process_progress.update(0, status='error')
                return (
                    no_update,
                    dbc.Alert(error_msg, color="danger"),
                    logs,
                    True,  # Disable interval
                    0,
                    "Error"
                )
            
            # Convert to DataFrame
            logger.set_progress(70)
            df = pd.DataFrame(entries)
            logs = logger.log_info("Converting to DataFrame...")
            
            # Count entries with meaningful abstracts for search
            logger.set_progress(80)
            abstract_count = 0
            searchable_count = 0
            if 'abstract' in df.columns:
                abstract_count = df['abstract'].notna().sum()
                # Count meaningful abstracts (same criteria as summary)
                meaningful_abstracts = (
                    df['abstract'].notna() & 
                    (df['abstract'].str.strip() != '') &
                    (df['abstract'].str.len() > 50) &
                    (~df['abstract'].str.lower().str.contains('no abstract|abstract not available|n/a', na=False))
                )
                searchable_count = meaningful_abstracts.sum()
                logs = logger.log_info(f"{abstract_count} entries have abstracts, {searchable_count} are searchable (50+ chars)")
            
            # Save the filtered data without tags
            logger.set_progress(90)
            logs = logger.log_info("Preparing data for analysis...")
            df_json = df.to_json(orient='split', date_format='iso')
            
            success_msg = f"Successfully filtered {len(df)} entries ({searchable_count} with searchable abstracts). Ready for search!"
            logs = logger.log_success(success_msg)
            
            logger.set_progress(100)
            process_progress.update(100, "Processing complete!", status='complete')
            return (
                df_json,
                dbc.Alert(success_msg, color="success"),
                logs,
                True,  # Disable interval
                100,
                "Complete!"
            )
            
        except Exception as e:
            error_msg = f"Error during filtering: {str(e)}"
            logs = logger.log_error(error_msg, e)
            process_progress.update(0, status='error')
            return (
                no_update,
                dbc.Alert(error_msg, color="danger"),
                logs,
                True,  # Disable interval
                0,
                "Error"
            )
    
    # Search functionality callbacks
    @app.callback(
        [Output('search-button', 'disabled'),
         Output('search-button-help', 'children')],
        [Input('search-query', 'value'),
         Input('data-store', 'data'),
         Input('upload-bibtex', 'contents'),
         Input('upload-bibtex', 'filename')]
    )
    def update_search_button(query, data, upload_contents, filename):
        """Enable search button when query is entered and data is available."""
        # Check if we have uploaded content or processed data
        has_data = bool(data) or bool(upload_contents) or bool(filename)
        
        if not has_data:
            return True, "Upload a bibliography file first to enable search"
        elif not query or not query.strip():
            return True, "Enter a search query (e.g., 'chronic fatigue syndrome')"
        else:
            # If we have a query and some form of data, enable the button
            return False, "Ready to search your uploaded bibliography"
    
    @app.callback(
        [Output('search-results-container', 'children'),
         Output('search-status', 'children'),
         Output('search-progress', 'value'),
         Output('search-progress-text', 'children'),
         Output('search-logs', 'children'),
         Output('search-progress-card', 'style'),
         Output('search-results-store', 'data'),
         Output('search-interval', 'disabled', allow_duplicate=True)],
        [Input('search-button', 'n_clicks')],
        [State('search-query', 'value'),
         State('search-methods', 'value'),
         State('semantic-threshold', 'value'),
         State('fuzzy-threshold', 'value'),
         State('max-results', 'value'),
         State('hybrid-model', 'value'),
         State('llm-threshold', 'value'),
         State('data-store', 'data'),
         State('upload-bibtex', 'filename')],
        prevent_initial_call=True
    )
    def perform_search(n_clicks, query, methods, semantic_threshold, fuzzy_threshold, max_results, hybrid_model, llm_threshold, data, filename):
        """Perform semantic search on the loaded data with detailed logging."""
        if not n_clicks or not query:
            return "", "", 0, "", "Ready to search...", {"display": "none"}, None, True
        
        # Reset and initialize progress tracker
        search_progress.__init__()  # Reset the tracker
        search_progress.update(0, "Starting search...", status='searching')
        
        # Initialize progress logger
        logger = ProgressLogger(search_progress)
        progress = 0
        
        # Show progress card
        progress_style = {"display": "block"}
        
        try:
            logs = logger.log_info(f"Starting search for: '{query}'")
            logger.set_progress(5)
            
            # Get data - either from store or by processing file directly
            if data:
                df = pd.read_json(data, orient='split')
                logs = logger.log_info(f"Loaded {len(df)} papers from data store")
            elif filename:
                # Fallback: process file directly if no filtered data available
                logs = logger.log_info(f"No processed data found, loading from file: {filename}")
                file_path = upload_dir / filename
                entries = process_bibtex_file(file_path)
                df = pd.DataFrame(entries)
                logs = logger.log_info(f"Loaded {len(df)} papers from file")
            else:
                error_msg = "No data available for search"
                logs = logger.log_error(error_msg)
                search_progress.update(0, status='error')
                return (
                    dbc.Alert(error_msg, color="danger"),
                    "",
                    0,
                    "Error",
                    logs,
                    progress_style,
                    None,
                    True  # Disable interval
                )
            
            # Update progress
            progress = 10
            logger.set_progress(progress)
            
            # Initialize searcher based on methods
            try:
                if 'hybrid' in methods or 'llm_only' in methods:
                    searcher = HybridSemanticSearcher(llm_model=hybrid_model)
                    if 'llm_only' in methods:
                        logs = logger.log_info(f"Initialized LLM-only searcher ({hybrid_model})")
                    else:
                        logs = logger.log_info(f"Initialized hybrid searcher (embeddings + {hybrid_model})")
                else:
                    # Get API key from environment
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key and 'semantic' in methods:
                        error_msg = "OpenAI API key required for semantic search. Please set OPENAI_API_KEY in your .env file."
                        logs = logger.log_error(error_msg)
                        search_progress.update(0, status='error')
                        return (
                            dbc.Alert(error_msg, color="danger"),
                            "",
                            0,
                            "Error",
                            logs,
                            progress_style,
                            None,
                            True  # Disable interval
                        )
                    searcher = SemanticSearcher(api_key=api_key) if api_key else SemanticSearcher()
                    logs = logger.log_info("Initialized standard semantic searcher")
            except Exception as e:
                error_msg = f"Failed to initialize searcher: {str(e)}"
                logs = logger.log_error(error_msg)
                search_progress.update(0, status='error')
                return (
                    dbc.Alert(error_msg, color="danger"),
                    "",
                    0,
                    "Error",
                    logs,
                    progress_style,
                    None,
                    True  # Disable interval
                )
            
            # Update progress
            progress = 20
            logger.set_progress(progress)
            
            # Perform search with logging
            logs = logger.log_info(f"Search methods: {', '.join(methods)}")
            logs = logger.log_info(f"Thresholds - Semantic: {semantic_threshold:.2f}, Fuzzy: {fuzzy_threshold}%")
            
            # Handle special search methods
            if 'llm_only' in methods:
                # LLM-only search - analyze ALL papers with GPT
                llm_results = searcher.llm_only_search(
                    query=query,
                    df=df,
                    max_results=max_results,
                    relevance_threshold=llm_threshold,
                    logger=logger
                )
                
                # Convert LLM results to DataFrame format
                if llm_results:
                    results_data = []
                    for idx, score in llm_results:
                        paper_data = df.iloc[idx].to_dict()
                        paper_data['search_score'] = score
                        paper_data['exact_score'] = 0
                        paper_data['fuzzy_score'] = 0
                        paper_data['semantic_score'] = 0
                        paper_data['llm_only_score'] = score
                        paper_data['original_index'] = idx
                        
                        # Add LLM analysis data if available
                        analyzed_papers = getattr(searcher, '_last_analyzed_papers', [])
                        llm_paper = next((p for p in analyzed_papers if p.get('original_index') == idx), None)
                        if llm_paper:
                            paper_data['llm_relevance_score'] = llm_paper.get('llm_relevance_score', 0)
                            paper_data['llm_confidence'] = llm_paper.get('llm_confidence', 0)
                            paper_data['llm_reasoning'] = llm_paper.get('llm_reasoning', '')
                            paper_data['llm_key_concepts'] = llm_paper.get('llm_key_concepts', [])
                        
                        results_data.append(paper_data)
                    
                    results = pd.DataFrame(results_data)
                else:
                    results = pd.DataFrame()
            elif 'hybrid' in methods:
                # Hybrid search - use specialized hybrid method
                hybrid_results = searcher.hybrid_search(
                    query=query,
                    df=df,
                    threshold=semantic_threshold,
                    max_embedding_candidates=50,
                    max_results=max_results,
                    logger=logger
                )
                
                # Get the LLM analyzed papers from the hybrid searcher
                analyzed_papers = getattr(searcher, '_last_analyzed_papers', [])
                
                # Convert hybrid results to DataFrame format with LLM analysis
                if hybrid_results:
                    results_data = []
                    for idx, score in hybrid_results:
                        paper_data = df.iloc[idx].to_dict()
                        paper_data['search_score'] = score
                        paper_data['exact_score'] = 0
                        paper_data['fuzzy_score'] = 0
                        paper_data['semantic_score'] = 0
                        paper_data['hybrid_score'] = score
                        paper_data['original_index'] = idx
                        
                        # Add LLM analysis data if available
                        llm_paper = next((p for p in analyzed_papers if p.get('original_index') == idx), None)
                        if llm_paper:
                            paper_data['llm_relevance_score'] = llm_paper.get('llm_relevance_score', 0)
                            paper_data['llm_confidence'] = llm_paper.get('llm_confidence', 0)
                            paper_data['llm_reasoning'] = llm_paper.get('llm_reasoning', '')
                            paper_data['llm_key_concepts'] = llm_paper.get('llm_key_concepts', [])
                        
                        results_data.append(paper_data)
                    
                    results = pd.DataFrame(results_data)
                else:
                    results = pd.DataFrame()
            else:
                # Standard multi-search
                results = searcher.multi_search(
                    query=query,
                    df=df,
                    methods=methods,
                    semantic_threshold=semantic_threshold,
                    fuzzy_threshold=fuzzy_threshold,
                    max_results=max_results,
                    logger=logger
                )
            
            # Update progress
            progress = 90
            logger.set_progress(progress)
            
            if results.empty:
                logs = logger.log_info("No results found with current settings")
                search_progress.update(100, "No results found", status='complete')
                return (
                    dbc.Alert(
                        f"No results found for '{query}' with the current settings. Try lowering the thresholds or using different search methods.",
                        color="warning"
                    ),
                    "",
                    100,
                    "Search complete - No results",
                    logs,
                    progress_style,
                    None,
                    True  # Disable interval
                )
            
            logs = logger.log_info(f"Formatting {len(results)} results for display...")
            
            # Create results display
            results_cards = []
            for idx, row in results.iterrows():
                # Create score badges
                score_badges = []
                if row.get('llm_only_score', 0) > 0:
                    # LLM-only search - show detailed LLM analysis
                    llm_score = row.get('llm_relevance_score', row['llm_only_score'] * 10)
                    confidence = row.get('llm_confidence', 8)
                    score_badges.append(dbc.Badge(f"🤖 LLM: {llm_score:.1f}/10 (confidence: {confidence:.1f})", color="dark", className="me-2"))
                    
                    # Add LLM reasoning
                    if 'llm_reasoning' in row and row['llm_reasoning']:
                        reasoning_badge = dbc.Badge(
                            f"💡 {row['llm_reasoning'][:60]}...", 
                            color="light", 
                            text_color="dark",
                            className="me-2 mb-1"
                        )
                        score_badges.append(reasoning_badge)
                    
                    # Add key concepts if available
                    if 'llm_key_concepts' in row and row['llm_key_concepts']:
                        concepts = row['llm_key_concepts'][:3]  # Show first 3 concepts
                        for concept in concepts:
                            score_badges.append(dbc.Badge(concept, color="secondary", className="me-1"))
                elif row.get('hybrid_score', 0) > 0:
                    # Hybrid search - show LLM analysis if available
                    score_badges.append(dbc.Badge(f"🚀 Hybrid: {row['hybrid_score']:.3f}", color="warning", className="me-2"))
                    
                    # Add LLM reasoning if available (from hybrid search)
                    if 'llm_reasoning' in row and row['llm_reasoning']:
                        reasoning_badge = dbc.Badge(
                            f"💡 {row['llm_reasoning'][:50]}...", 
                            color="light", 
                            text_color="dark",
                            className="me-2 mb-1"
                        )
                        score_badges.append(reasoning_badge)
                else:
                    # Standard search badges
                    if row.get('exact_score', 0) > 0:
                        score_badges.append(dbc.Badge(f"Exact: {row['exact_score']:.2f}", color="success", className="me-2"))
                    if row.get('fuzzy_score', 0) > 0:
                        score_badges.append(dbc.Badge(f"Fuzzy: {row['fuzzy_score']:.2f}", color="info", className="me-2"))
                    if row.get('semantic_score', 0) > 0:
                        score_badges.append(dbc.Badge(f"Semantic: {row['semantic_score']:.2f}", color="primary", className="me-2"))
                
                # Create paper card
                card = dbc.Card([
                    dbc.CardHeader([
                        html.H5(row.get('title', 'No title'), className="mb-1"),
                        html.Small(f"Overall Score: {row['search_score']:.3f}", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Authors: "), 
                            row.get('author', 'No author')
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Year: "), 
                            str(row.get('year', 'No year'))
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Publication ID: "), 
                            get_field_case_insensitive(row, 'publication_id') or 'No ID'
                        ], className="mb-2") if get_field_case_insensitive(row, 'publication_id') else None,
                        html.P([
                            html.Strong("Journal: "), 
                            row.get('journal', 'No journal')
                        ], className="mb-2") if row.get('journal') else None,
                        html.P([
                            html.Strong("Abstract: "), 
                            (row.get('abstract', '')[:300] + '...' if len(str(row.get('abstract', ''))) > 300 else row.get('abstract', 'No abstract'))
                        ], className="mb-2"),
                        html.Div(score_badges) if score_badges else None
                    ])
                ], className="mb-3")
                
                results_cards.append(card)
            
            # Complete progress
            progress = 100
            logger.set_progress(progress)
            logs = logger.log_success(f"Search display ready! Showing {len(results)} results")
            search_progress.update(100, "Search complete!", status='complete')
            
            # Store search results for download
            search_results_data = results.to_json(orient='split')
            
            return (
                html.Div([
                    dbc.Alert(
                        f"Found {len(results)} results for '{query}'",
                        color="success",
                        className="mb-3"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "📄 Download CSV",
                                id="download-search-csv-button",
                                color="primary",
                                size="sm",
                                className="mb-3"
                            )
                        ], width="auto")
                    ]),
                    html.Div(results_cards)
                ]),
                "",
                progress,
                "Search complete!",
                logs,
                progress_style,
                search_results_data,
                True  # Disable interval
            )
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            if 'logger' in locals():
                logs = logger.log_error(error_msg, e)
            else:
                logs = f"[ERROR] {error_msg}"
            
            search_progress.update(0, status='error')
            return (
                dbc.Alert(error_msg, color="danger"),
                "",
                0,
                "Error occurred",
                logs,
                progress_style,
                None,
                True  # Disable interval
            )

    @app.callback(
        Output('download-search-results', 'data'),
        [Input('download-search-csv-button', 'n_clicks')],
        [State('search-results-store', 'data'),
         State('search-query', 'value')],
        prevent_initial_call=True
    )
    def download_search_results_csv(n_clicks, search_results_data, query):
        """Download search results as CSV file."""
        if not n_clicks or not search_results_data:
            raise PreventUpdate
        
        try:
            # Load search results from store
            results_df = pd.read_json(search_results_data, orient='split')
            
            if results_df.empty:
                raise PreventUpdate
            
            # Select key columns for CSV export
            export_columns = []
            
            # Define desired columns in order
            desired_columns = [
                'publication_id', 'title', 'year', 'author', 'journal', 
                'doi', 'abstract', 'search_score', 'tags'
            ]
            
            # Add columns that exist in the dataframe
            for col in desired_columns:
                if col in results_df.columns:
                    export_columns.append(col)
                elif col == 'publication_id':
                    # Try case-insensitive match for publication_id
                    for df_col in results_df.columns:
                        if df_col.lower() == 'publication_id':
                            results_df['publication_id'] = results_df[df_col]
                            export_columns.append('publication_id')
                            break
            
            # Add any LLM analysis columns if they exist
            llm_columns = [
                'llm_relevance_score', 'llm_confidence', 'llm_reasoning', 'llm_key_concepts',
                'exact_score', 'fuzzy_score', 'semantic_score', 'hybrid_score', 'llm_only_score'
            ]
            
            for col in llm_columns:
                if col in results_df.columns and col not in export_columns:
                    export_columns.append(col)
            
            # Select only the export columns
            export_df = results_df[export_columns]
            
            # Clean query for filename
            clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_query = clean_query.replace(' ', '_')[:30]  # Limit filename length
            
            filename = f"search_results_{clean_query}_{len(export_df)}papers.csv"
            
            return dcc.send_data_frame(export_df.to_csv, filename=filename, index=False)
            
        except Exception as e:
            # If there's an error, we can't show it in the download callback
            # So we'll just prevent the update
            raise PreventUpdate

    @app.callback(
        Output('wordcloud-collapse', 'is_open'),
        [Input('wordcloud-toggle', 'n_clicks')],
        [State('wordcloud-collapse', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_wordcloud_options(n_clicks, is_open):
        """Toggle word cloud options visibility."""
        if n_clicks:
            return not is_open
        return is_open
    
    # Interval callback for updating search progress
    @app.callback(
        [Output('search-progress', 'value', allow_duplicate=True),
         Output('search-progress-text', 'children', allow_duplicate=True),
         Output('search-logs', 'children', allow_duplicate=True),
         Output('search-interval', 'disabled')],
        [Input('search-interval', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_search_progress_display(n_intervals):
        """Update search progress display from the global tracker."""
        state = search_progress.get_state()
        
        # Disable interval if search is complete or idle
        disable_interval = state['status'] in ['idle', 'complete', 'error']
        
        return (
            state['progress'],
            f"Progress: {state['progress']}%",
            state['logs'],
            disable_interval
        )
    
    # Interval callback for updating processing progress
    @app.callback(
        [Output('process-progress', 'value', allow_duplicate=True),
         Output('process-progress-text', 'children', allow_duplicate=True),
         Output('processing-logs', 'children', allow_duplicate=True),
         Output('process-interval', 'disabled')],
        [Input('process-interval', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_process_progress_display(n_intervals):
        """Update processing progress display from the global tracker."""
        state = process_progress.get_state()
        
        # Disable interval if processing is complete or idle
        disable_interval = state['status'] in ['idle', 'complete', 'error']
        
        return (
            state['progress'],
            f"Progress: {state['progress']}%",
            state['logs'],
            disable_interval
        )
    
    # Callback to enable processing interval when filter button is clicked
    @app.callback(
        Output('process-interval', 'disabled', allow_duplicate=True),
        [Input('filter-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def enable_process_interval_filter(n_clicks):
        """Enable the processing interval when filter button is clicked."""
        if n_clicks:
            return False  # Enable interval
        return True  # Keep disabled
    
    # Callback to enable processing interval when process button is clicked
    @app.callback(
        Output('process-interval', 'disabled', allow_duplicate=True),
        [Input('process-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def enable_process_interval_process(n_clicks):
        """Enable the processing interval when process button is clicked."""
        if n_clicks:
            return False  # Enable interval
        return True  # Keep disabled


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
