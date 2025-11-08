"""Interactive dashboard for Bibtex Analyzer."""
import base64
import io
import json
import os
import random
import math
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
from datetime import datetime
import matplotlib

# Use the 'Agg' backend to avoid GUI threading issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .bibtex_processor import process_bibtex_file, BibtexProcessor
import bibtexparser
from .tag_generator import TagGenerator
from .semantic_search import SemanticSearcher, HybridSemanticSearcher
from .staff_analyzer import StaffAnalyzer, StaffLLMSummarizer

UPLOAD_DIR = Path("uploads")
MANIFEST_PATH = UPLOAD_DIR / "manifest.json"
EMBEDDING_DIR = Path(".dataset_embeddings")

UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDING_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_manifest() -> List[Dict[str, Any]]:
    """Load persisted dataset metadata."""
    if not MANIFEST_PATH.exists():
        return []
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            if isinstance(manifest, list):
                return manifest
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_dataset_manifest(manifest: List[Dict[str, Any]]) -> None:
    """Persist dataset manifest to disk."""
    try:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        # Failing to write manifest should not crash the app; log to stdout.
        print("Warning: Unable to write dataset manifest.")


def add_manifest_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Add or replace a dataset entry in the manifest."""
    manifest = load_dataset_manifest()
    manifest = [existing for existing in manifest if existing.get("id") != entry.get("id")]
    manifest.append(entry)
    # Sort newest first by uploaded_at
    manifest.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
    save_dataset_manifest(manifest)
    return manifest


def format_dataset_label(entry: Dict[str, Any]) -> str:
    """Format a manifest entry into a human readable dropdown label."""
    uploaded_at = entry.get("uploaded_at")
    label_time = uploaded_at
    if uploaded_at:
        try:
            dt = datetime.fromisoformat(uploaded_at)
            label_time = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            label_time = uploaded_at
    original_name = entry.get("original_name", entry.get("stored_name", "Dataset"))
    total_entries = entry.get("entry_count")
    size_hint = f" · {total_entries} rows" if isinstance(total_entries, int) else ""
    return f"{original_name} · {label_time}{size_hint}"


def get_manifest_entry(manifest: List[Dict[str, Any]], dataset_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return manifest entry by dataset id."""
    if not dataset_id:
        return None
    for entry in manifest:
        if entry.get("id") == dataset_id:
            return entry
    return None


def ensure_embeddings_for_dataset(dataset_id: str, df: pd.DataFrame, logger: Optional["DashLogger"] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Ensure embeddings exist for a dataset and return metadata."""
    metadata: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "row_count": int(len(df)),
        "status": "unavailable",
    }

    if df.empty:
        metadata["status"] = "empty"
        return None, metadata

    embedding_path = EMBEDDING_DIR / f"{dataset_id}.npz"

    def _load_embeddings(path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            embeddings = data["embeddings"]
            row_index = data["row_index"]
            expected_index = np.arange(len(df))
            if embeddings.shape[0] == len(df) and np.array_equal(row_index, expected_index):
                return embeddings
        except Exception as exc:  # pylint: disable=broad-except
            if logger:
                logger.log_error(f"Failed to load saved embeddings: {exc}")
        return None

    embeddings = _load_embeddings(embedding_path)
    if embeddings is not None:
        metadata["status"] = "loaded"
        metadata["path"] = str(embedding_path)
        return embeddings, metadata

    try:
        searcher = SemanticSearcher()
    except ValueError as exc:
        if logger:
            logger.log_error(f"Cannot compute embeddings without API key: {exc}")
        metadata["status"] = "missing_api_key"
        return None, metadata

    if logger:
        logger.log_info("Computing embeddings for selected dataset...")

    paper_texts = [searcher.prepare_paper_text(row.to_dict()) for _, row in df.iterrows()]
    embeddings = searcher.get_embeddings_batch(paper_texts, logger=logger)

    try:
        np.savez_compressed(
            embedding_path,
            embeddings=embeddings,
            row_index=np.arange(len(df)),
        )
        metadata["status"] = "created"
        metadata["path"] = str(embedding_path)
    except Exception as exc:  # pylint: disable=broad-except
        if logger:
            logger.log_error(f"Failed to persist embeddings: {exc}")
        metadata["status"] = "computed_in_memory"

    return embeddings, metadata


def generate_bibliography_summary(file_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Generate summary content for a bibliography file."""
    if not file_path.exists():
        error_msg = f"File '{file_path.name}' not found in uploads."
        return dbc.Alert(error_msg, color="warning"), {"display": "block"}

    processor = BibtexProcessor()
    entries = processor.load_entries(str(file_path), deduplicate=True)
    df = pd.DataFrame(entries)
    dedup_stats = processor.get_deduplication_stats()

    total_entries = len(df)
    if total_entries == 0:
        return dbc.Alert("No entries detected in the selected dataset.", color="warning"), {"display": "block"}

    has_title = 0
    has_abstract = 0
    has_author = 0
    has_year = 0
    has_journal = 0
    has_doi = 0

    if 'title' in df.columns:
        meaningful_titles = df['title'].dropna().str.strip().str.len() > 10
        has_title = meaningful_titles.sum()

    if 'abstract' in df.columns:
        meaningful_abstracts = (
            df['abstract'].notna() &
            (df['abstract'].str.strip() != '') &
            (df['abstract'].str.len() > 50) &
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
        meaningful_dois = (
            df['doi'].notna() &
            (df['doi'].str.strip() != '') &
            df['doi'].str.contains(r'10\.', na=False)
        )
        has_doi = meaningful_dois.sum()

    year_range = "N/A"
    if 'year' in df.columns and has_year > 0:
        years = pd.to_numeric(df['year'], errors='coerce')
        valid_years = years[(years >= 1000) & (years <= 2100)].dropna()
        if len(valid_years) > 0:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())
            year_range = f"{min_year} - {max_year}"

    summary_content = [
        dbc.Row([
            dbc.Col([
                html.H6("📄 Total Entries", className="text-primary mb-1"),
                html.H4(f"{total_entries:,}", className="mb-0"),
                html.Small(
                    f"({dedup_stats['total_entries']:,} original)" if dedup_stats['duplicates_removed'] > 0 else "",
                    className="text-muted"
                )
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
        html.H6("🔍 Search Readiness:", className="mb-2"),
        html.Div([
            dbc.Badge(
                f"{has_abstract:,} papers ready for semantic search",
                color="success" if has_abstract > total_entries * 0.5 else "warning",
                className="me-2 mb-1"
            ),
            dbc.Badge(
                f"{has_year:,} with valid years for filtering",
                color="info",
                className="me-2 mb-1"
            ),
            dbc.Badge(
                f"{total_entries - has_abstract:,} missing meaningful abstracts",
                color="secondary",
                className="me-2 mb-1"
            ),
        ]),
    ]

    return summary_content, {"display": "block"}


PAGE_SIZE = 20


def _compute_page_slice(df: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return df.iloc[start:end]


def format_pagination_text(page: int, page_size: int, total: int) -> str:
    if total == 0:
        return "No results to display."
    start = (page - 1) * page_size + 1
    end = min(page * page_size, total)
    total_pages = max(1, math.ceil(total / page_size))
    return f"Showing {start}-{end} of {total} (Page {page} of {total_pages})"


def create_result_cards(df: pd.DataFrame, page: int, page_size: int) -> List[Any]:
    if df is None or df.empty:
        return [html.Div("No results to display.", className="text-muted")]

    subset = _compute_page_slice(df, page, page_size)
    cards: List[Any] = []

    for _, row in subset.iterrows():
        score_badges = []
        if row.get('llm_only_score', 0) > 0:
            llm_score = row.get('llm_relevance_score', row['llm_only_score'] * 10)
            confidence = row.get('llm_confidence', 0)
            score_badges.append(dbc.Badge(f'LLM Score: {llm_score:.1f}/10', color='primary', className='me-2 mb-2'))
            score_badges.append(dbc.Badge(f'Confidence: {confidence:.1f}/10', color='info', className='mb-2'))
        else:
            exact_score = row.get('exact_score') or 0
            fuzzy_score = row.get('fuzzy_score') or 0
            semantic_score = row.get('semantic_score') or 0
            hybrid_score = row.get('hybrid_score') or 0
            if exact_score > 0:
                score_badges.append(dbc.Badge(f"Exact: {exact_score:.3f}", color='success', className='me-2 mb-2'))
            if fuzzy_score > 0:
                score_badges.append(dbc.Badge(f"Fuzzy: {fuzzy_score:.1f}", color='warning', className='me-2 mb-2'))
            if semantic_score > 0:
                score_badges.append(dbc.Badge(f"Semantic: {semantic_score:.3f}", color='primary', className='me-2 mb-2'))
            if hybrid_score > 0:
                score_badges.append(dbc.Badge(f"Hybrid: {hybrid_score:.3f}", color='secondary', className='me-2 mb-2'))

        paper_url = row.get('url')
        has_url = isinstance(paper_url, str) and paper_url.strip()
        title_text = row.get('title', 'No title')
        title_component = html.A(
            title_text,
            href=paper_url,
            target="_blank",
            className="text-decoration-none fw-semibold"
        ) if has_url else html.Span(title_text, className="fw-semibold")

        score_text = row.get('search_score', 0)

        card_body_children = [
            html.P([html.Strong('Authors: '), row.get('author', 'No author')], className='mb-2'),
            html.P([html.Strong('Year: '), str(row.get('year', 'No year'))], className='mb-2'),
            html.P([html.Strong('Journal: '), row.get('journal', 'No journal')], className='mb-2') if row.get('journal') else None,
            html.P([
                html.Strong('Abstract: '),
                (row.get('abstract', '')[:300] + '...' if len(str(row.get('abstract', ''))) > 300 else row.get('abstract', 'No abstract'))
            ], className='mb-2'),
            html.Div(score_badges) if score_badges else None
        ]

        if has_url:
            card_body_children.append(
                dbc.Button(
                    "View Paper",
                    href=paper_url,
                    target="_blank",
                    color="primary",
                    size="sm",
                    className="mt-2",
                    external_link=True
                )
            )

        card = dbc.Card([
            dbc.CardHeader([
                title_component,
                html.Small(f" Overall Score: {score_text:.3f}", className='text-muted ms-2')
            ]),
            dbc.CardBody([child for child in card_body_children if child is not None])
        ], className='mb-3')

        cards.append(card)

    return cards


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
    
    initial_manifest = load_dataset_manifest()
    dataset_options = [
        {"label": format_dataset_label(entry), "value": entry["id"]}
        for entry in initial_manifest
        if entry.get("id")
    ]
    initial_dataset_value = dataset_options[0]["value"] if dataset_options else None
    
    # Add initial store for tracking first load
    app.layout = html.Div([
        # Stores
        dcc.Store(id='initial-load', data=True),
        dcc.Store(id='dataset-manifest-store', data=initial_manifest),
        dcc.Store(id='embedding-store'),
        dcc.Store(id='search-params-store'),
        dcc.Store(id='data-store'),
        dcc.Store(id='tagged-data-store'),
        dcc.Store(id='wordcloud-store'),
        dcc.Store(id='wordcloud-html-store', data=None),
        dcc.Store(id='search-results-store'),
        dcc.Store(id='staff-analysis-store'),
        dcc.Store(id='staff-publication-map', data={}),
        dcc.Store(id='search-summary-store', data=None),
        dcc.Store(id='search-progress-store', data={'progress': 0, 'logs': '', 'status': 'idle'}),
        dcc.Store(id='process-progress-store', data={'progress': 0, 'logs': '', 'status': 'idle'}),
        dcc.Store(id='search-pagination-store', data={'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0}),
        dcc.Store(id='results-visibility-store', data={'show': False}),
        dcc.Store(id='upload-timestamp', data=None),  # Track upload time to force updates
        dcc.Interval(id='search-interval', interval=500, disabled=True),  # Update every 500ms during search
        dcc.Interval(id='process-interval', interval=500, disabled=True),  # Update every 500ms during processing
        dcc.Download(id="download-wordcloud"),
        dcc.Download(id="download-tagged-data"),
        dcc.Download(id="download-search-results"),
        dcc.Download(id="download-staff-analysis"),
        
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
                    dbc.Card([
                        dbc.CardHeader("📦 Saved Datasets"),
                        dbc.CardBody([
                            dbc.Label("Previously Uploaded Files", html_for="dataset-selector"),
                            dcc.Dropdown(
                                id='dataset-selector',
                                options=dataset_options,
                                value=initial_dataset_value,
                                placeholder="Select a dataset to load",
                                clearable=True,
                                className="mb-2"
                            ),
                            dbc.FormText(
                                "Select a dataset to reload it without uploading. Embeddings reload automatically when available."
                            ),
                            html.Div(id="embedding-status", className="mt-2 small text-muted"),
                        ])
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
                                            html.Strong("Pipeline"), ": Exact → Fuzzy → Semantic build the candidate pool; ",
                                            html.Strong("Hybrid"), " reranks those matches with GPT for relevance; ",
                                            html.Strong("LLM Only"), " lets GPT score every paper (costly but comprehensive)."
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
                                            value=None,
                                            placeholder="No limit",
                                            className="mb-1"
                                        ),
                                        dbc.FormText(
                                            "Optional: limit on-screen results; downloads always include every match.",
                                            className="mb-3"
                                        )
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
                    
                    html.Div(id="search-summary-card", className="mb-4"),

                    # Staff Analysis Container
                    html.Div(id="staff-analysis-container", className="mb-4"),
                    html.Div([
                        dbc.Button(
                            "Generate LLM Staff Summaries",
                            id="generate-staff-summaries",
                            color="secondary",
                            className="me-3 mb-2",
                            disabled=True
                        ),
                        dcc.Loading(
                            type="circle",
                            children=html.Div(
                                "Run a search to enable LLM summaries.",
                                id="staff-summary-status",
                                className="small text-muted mb-2"
                            )
                        )
                    ], className="mb-4"),

                    # Search Results Controls
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Show Results",
                                    id="toggle-results-btn",
                                    color="primary",
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "⬇️ Download Results",
                                    id="download-search-csv-button",
                                    color="secondary",
                                    className="me-2",
                                    disabled=True
                                ),
                                dbc.Button(
                                    "⬇️ Download Staff Summary",
                                    id="download-staff-summary-button",
                                    color="secondary",
                                    disabled=True
                                ),
                            ], md="auto", className="mb-2"),
                            dbc.Col([
                                dbc.Button(
                                    "◀ Previous",
                                    id="search-prev-page",
                                    color="link",
                                    className="px-0 me-2",
                                    disabled=True
                                ),
                                dbc.Button(
                                    "Next ▶",
                                    id="search-next-page",
                                    color="link",
                                    className="px-0",
                                    disabled=True
                                ),
                            ], width="auto", className="mb-2"),
                            dbc.Col(id="pagination-info", className="text-end small text-muted"),
                        ], className="align-items-center g-2"),
                        html.Div(
                            "CSV downloads include all deduplicated matches, regardless of the Max Results setting.",
                            className="text-muted small mb-2"
                        ),
                    ], className="mb-3"),

                    # Search Results Container
                    html.Div(id="search-results-container", className="mt-3"),
                    
                    # Status and Result Display
                    html.Div(id="search-status"),
                ], width=12)
            ]),
            
            # Modal for staff publications
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id='staff-pubs-modal-title')),
                    dbc.ModalBody(id='staff-pubs-content'),
                    dbc.ModalFooter(
                        dbc.Button("Close", id='staff-pubs-close', color='secondary')
                    ),
                ],
                id='staff-pubs-modal',
                size='lg',
                is_open=False,
                scrollable=True,
                centered=True
            ),
            
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

def create_staff_analysis_ui(staff_df: pd.DataFrame, query: str) -> html.Div:
    """Create UI component for staff analysis results.
    
    Args:
        staff_df: DataFrame with staff metrics
        query: Search query used
        
    Returns:
        Dash HTML component with staff analysis
    """
    if staff_df.empty:
        return html.Div()
    
    # Create summary cards for top staff
    top_staff = staff_df.head(10)
    
    staff_cards = []
    for idx, row in top_staff.iterrows():
        # Determine tier and badge color
        tier = StaffAnalyzer().get_staff_tier(row.to_dict())
        tier_colors = {
            "Star Performer": "success",
            "Rising Star": "info",
            "Prolific Contributor": "warning",
            "Developing Researcher": "secondary"
        }
        
        # Create data quality indicator
        completeness = row['avg_completeness']
        if completeness >= 0.8:
            quality_icon = "🟢"
        elif completeness >= 0.4:
            quality_icon = "🟡"
        else:
            quality_icon = "🔴"
        
        card = dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H6(f"#{row['rank']} - {row['staff_id']}", className="mb-0"),
                    dbc.Badge(tier, color=tier_colors.get(tier, "secondary"), className="ms-2")
                ], className="d-flex align-items-center")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("Publications: "),
                            f"{row['publication_count']}"
                        ], className="mb-1"),
                        html.P([
                            html.Strong("Impact Score: "),
                            f"{row['impact_score']:.3f}"
                        ], className="mb-1"),
                    ], width=6),
                    dbc.Col([
                        html.P([
                            html.Strong("Avg Relevance: "),
                            f"{row['avg_relevance']:.3f}"
                        ], className="mb-1"),
                        html.P([
                            html.Strong("Total Citations: "),
                            f"{int(row['total_citations']):,}"
                        ], className="mb-1"),
                    ], width=6),
                ]),
                html.Hr(className="my-2"),
                html.Small([
                    f"{quality_icon} Data Quality: {completeness:.0%} | ",
                    f"Active Years: {row['year_range']}"
                ], className="text-muted")
            ])
        ], className="mb-3")
        
        staff_cards.append(card)
    
    # Create the main component
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4(f"🎯 Staff Analysis for '{query}'", className="mb-0"),
                html.Small(f"Found {len(staff_df)} staff members with matching publications", className="text-muted")
            ]),
            dbc.CardBody([
                # Summary statistics
                dbc.Row([
                    dbc.Col([
                        dbc.Alert([
                            html.H6("Top Performers", className="alert-heading"),
                            html.P(f"{len(staff_df[staff_df['publication_count'] >= 5])} staff with 5+ publications", className="mb-0")
                        ], color="success"),
                    ], width=4),
                    dbc.Col([
                        dbc.Alert([
                            html.H6("High Quality", className="alert-heading"),
                            html.P(f"{len(staff_df[staff_df['avg_quality'] >= 0.6])} staff with quality score ≥0.6", className="mb-0")
                        ], color="info"),
                    ], width=4),
                    dbc.Col([
                        dbc.Alert([
                            html.H6("Total Impact", className="alert-heading"),
                            html.P(f"{int(staff_df['total_citations'].sum()):,} total citations", className="mb-0")
                        ], color="primary"),
                    ], width=4),
                ]),
                
                # Top staff cards
                html.H5("Top 10 Staff by Impact Score", className="mt-4 mb-3"),
                html.Div(staff_cards),
            ])
        ])
    ])

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

def handle_upload_status(contents: Optional[str], filename: Optional[str], upload_dir: Path) -> Tuple[str, bool, bool, Any, List[Dict[str, Any]]]:
    """Persist uploaded file and refresh manifest information."""
    if contents is None or not filename:
        raise PreventUpdate

    manifest_before = load_dataset_manifest()

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
    except (ValueError, AttributeError, base64.binascii.Error) as exc:
        alert = dbc.Alert(f"Failed to decode upload: {exc}", color="danger")
        return "", True, True, alert, manifest_before

    dataset_id = uuid.uuid4().hex
    safe_name = Path(filename).name
    stored_name = f"{dataset_id}__{safe_name}"
    file_path = upload_dir / stored_name

    try:
        with open(file_path, 'wb') as f:
            f.write(decoded)
    except OSError as exc:
        alert = dbc.Alert(f"Failed to save upload: {exc}", color="danger")
        return "", True, True, alert, manifest_before

    entry_count = None
    try:
        processor = BibtexProcessor()
        entry_count = len(processor.load_entries(str(file_path)))
    except Exception:
        entry_count = None

    entry = {
        "id": dataset_id,
        "stored_name": stored_name,
        "original_name": safe_name,
        "uploaded_at": datetime.utcnow().isoformat(),
        "entry_count": entry_count,
        "content_type": content_type,
        "file_size": len(decoded),
    }
    manifest = add_manifest_entry(entry)

    alert = dbc.Alert(f"File '{safe_name}' uploaded and saved for reuse.", color="success")
    return (
        f"Selected: {safe_name}",
        False,
        False,
        alert,
        manifest
    )

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
        
        # Process BibTeX entries with deduplication
        df, error_msg, dedup_stats = process_bibtex_entries(file_path, logger, deduplicate=True)
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
        
        # Create success message with deduplication info
        status_msg = f"Successfully processed {total_count} entries"
        if dedup_stats and dedup_stats['duplicates_removed'] > 0:
            status_msg += f" ({dedup_stats['total_entries']} original → {dedup_stats['unique_entries']} unique)"
        status_msg += f", {tagged_count} with tags ({percentage}%)."
        
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
    columns_to_show = ['year', 'title', 'authors', 'journal', 'publication_id', 'doi', 'staff_id', 'all_staff_ids']
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
    
    # Format all_staff_ids for display
    if 'all_staff_ids' in df.columns:
        df['all_staff_ids'] = df['all_staff_ids'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x else ''
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

def process_bibtex_entries(file_path: Path, logger: 'DashLogger', deduplicate: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Dict]]:
    """Process bibliography entries from a BibTeX or CSV file with optional deduplication.
    
    Args:
        file_path: Path to the BibTeX or CSV file
        logger: Logger for tracking progress
        deduplicate: Whether to deduplicate entries
        
    Returns:
        DataFrame of bibliography entries, any error message, and deduplication stats
    """
    try:
        # Initialize processor
        processor = BibtexProcessor()
        
        # Detect file format and read accordingly
        file_ext = file_path.suffix.lower()
        logger.log_info(f"Reading {file_ext.upper().replace('.', '')} file: {file_path.name}")
        
        # Use BibtexProcessor to load entries with deduplication
        logger.log_info("Loading entries...")
        entries = processor.load_entries(str(file_path), deduplicate=deduplicate)
        
        if deduplicate:
            stats = processor.get_deduplication_stats()
            logger.log_info(f"Deduplication complete: {stats['total_entries']} total entries → {stats['unique_entries']} unique entries")
            logger.log_info(f"Removed {stats['duplicates_removed']} duplicates")
            
            # Log staff contributor information if available
            staff_count = len([e for e in entries if 'all_staff_ids' in e and len(e.get('all_staff_ids', [])) > 1])
            if staff_count > 0:
                logger.log_info(f"{staff_count} publications have multiple staff contributors")
        
        if not entries:
            error_msg = f"No entries found in the {file_ext.upper().replace('.', '')} file"
            logger.log_error(error_msg)
            return None, error_msg, None
            
        # Convert to DataFrame
        logger.log_info("Converting entries to DataFrame...")
        df = pd.DataFrame(entries)
        
        # Filter out entries without abstracts if needed
        if 'abstract' in df.columns:
            has_abstract = df['abstract'].notna() & (df['abstract'] != '')
            abstract_count = has_abstract.sum()
            logger.log_info(f"Found {abstract_count} entries with abstracts")
        
        logger.log_success(f"Successfully processed {len(df)} entries")
        return df, None, processor.get_deduplication_stats() if deduplicate else None
        
    except Exception as e:
        error_msg = "Failed to process BibTeX file"
        logger.log_error(error_msg, e)
        return None, error_msg, None

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
         Output('upload-timestamp', 'data'),
         Output('dataset-manifest-store', 'data', allow_duplicate=True)],
        [Input('upload-bibtex', 'contents')],
        [State('upload-bibtex', 'filename')],
        prevent_initial_call='initial_duplicate'
    )
    def update_upload_status(contents, filename):
        """Update the upload status and enable/disable the process button."""
        filename_text, process_disabled, filter_disabled, status_children, manifest = handle_upload_status(
            contents, filename, upload_dir
        )
        timestamp = datetime.now().isoformat() if contents else None
        return (
            filename_text,
            process_disabled,
            filter_disabled,
            status_children,
            timestamp,
            manifest,
        )

    @app.callback(
        [Output('dataset-selector', 'options'),
         Output('dataset-selector', 'value')],
        [Input('dataset-manifest-store', 'data')],
        [State('dataset-selector', 'value')],
        prevent_initial_call=False
    )
    def refresh_dataset_options(manifest, current_value):
        """Update dataset dropdown options whenever the manifest changes."""
        manifest = manifest or []
        options = [
            {"label": format_dataset_label(entry), "value": entry["id"]}
            for entry in manifest
            if entry.get("id")
        ]
        if not options:
            return [], None
        manifest_ids = [opt["value"] for opt in options]
        if current_value in manifest_ids:
            return options, current_value
        return options, manifest_ids[0]

    @app.callback(
        [Output('upload-filename', 'children', allow_duplicate=True),
         Output('process-button', 'disabled', allow_duplicate=True),
         Output('filter-button', 'disabled', allow_duplicate=True),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('data-store', 'data', allow_duplicate=True),
         Output('tagged-data-store', 'data', allow_duplicate=True),
         Output('embedding-store', 'data')],
        [Input('dataset-selector', 'value')],
        [State('dataset-manifest-store', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def load_dataset_from_selection(dataset_id, manifest):
        """Load dataset data and embeddings when a saved dataset is selected."""
        if not dataset_id:
            return (
                no_update,
                True,
                True,
                no_update,
                no_update,
                no_update,
                None,
            )

        manifest = manifest or []
        entry = get_manifest_entry(manifest, dataset_id)
        if not entry:
            alert = dbc.Alert("Selected dataset metadata not found.", color="warning")
            return (
                "",
                True,
                True,
                alert,
                None,
                None,
                {"dataset_id": dataset_id, "status": "missing_manifest"},
            )

        file_path = upload_dir / entry['stored_name']
        if not file_path.exists():
            alert = dbc.Alert(
                f"Dataset '{entry.get('original_name', entry['stored_name'])}' is missing on disk.",
                color="danger"
            )
            return (
                "",
                True,
                True,
                alert,
                None,
                None,
                {"dataset_id": dataset_id, "status": "missing_file"},
            )

        try:
            entries = process_bibtex_file(str(file_path))
        except Exception as exc:  # pylint: disable=broad-except
            alert = dbc.Alert(f"Failed to load dataset: {exc}", color="danger")
            return (
                "",
                True,
                True,
                alert,
                None,
                None,
                {"dataset_id": dataset_id, "status": "load_error"},
            )

        df = pd.DataFrame(entries).reset_index(drop=True)
        data_json = df.to_json(orient='split', date_format='iso') if not df.empty else pd.DataFrame().to_json(orient='split')

        logger = DashLogger()
        embeddings, embedding_metadata = ensure_embeddings_for_dataset(dataset_id, df, logger)

        status_color = "success" if embedding_metadata.get("status") in {"loaded", "created"} else "info"
        message_suffix = ""
        if embedding_metadata.get("status") == "missing_api_key":
            status_color = "warning"
            message_suffix = " (embeddings require an OpenAI API key to be set)"
        elif embedding_metadata.get("status") == "missing_file":
            status_color = "danger"
            message_suffix = " (embedding file missing; will recompute during search)"

        alert = dbc.Alert(
            f"Loaded dataset '{entry.get('original_name', entry['stored_name'])}'{message_suffix}.",
            color=status_color,
            className="mb-0"
        )

        embedding_metadata["dataset_id"] = dataset_id

        return (
            f"Loaded: {entry.get('original_name', entry['stored_name'])}",
            False,
            False,
            alert,
            data_json,
            None,
            embedding_metadata,
        )
    
    @app.callback(
        Output("embedding-status", "children"),
        [Input("embedding-store", "data")],
        [State("dataset-selector", "value")]
    )
    def update_embedding_status(metadata, dataset_id):
        """Display embedding readiness for the active dataset."""
        if not dataset_id:
            return html.Span("Select a dataset to prepare embeddings.", className="text-muted")
        
        if not metadata or metadata.get("dataset_id") != dataset_id:
            return html.Span("Embeddings pending…", className="text-warning")
        
        status = metadata.get("status", "unknown")
        row_count = metadata.get("row_count")
        path = metadata.get("path")
        
        status_map = {
            "loaded": ("Embeddings ready", "success"),
            "created": ("Embeddings generated", "success"),
            "computed_in_memory": ("Embeddings ready (not cached)", "warning"),
            "missing_api_key": ("API key required for embeddings", "danger"),
            "empty": ("No rows to embed", "secondary"),
            "unavailable": ("Embeddings unavailable", "secondary"),
            "no_dataset": ("Embeddings pending…", "warning"),
            "unknown": ("Embedding status unknown", "secondary"),
        }
        
        label, color = status_map.get(status, status_map["unknown"])
        details = []
        if isinstance(row_count, int) and row_count >= 0:
            details.append(f"{row_count} rows")
        if status == "loaded" and path:
            details.append("cached")
        
        message = " • ".join(details) if details else ""
        
        return html.Span([
            dbc.Badge(label, color=color, className="me-2"),
            html.Span(message, className="text-muted") if message else None
        ])
    
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
        
        # Helper to funnel log messages through the shared logger and return display text
        def update_log(message: str, level: str = "info") -> str:
            if level == "error":
                return logger.log_error(message)
            if level == "success":
                return logger.log_success(message)
            return logger.log_info(message)
        
        # Enable the interval immediately to show progress updates
        # This will be updated by the interval callback
        
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
            
            # Mark processing as complete
            process_progress.update(100, "Processing complete!", status='complete')
            
            return (
                df_json,
                df_json,
                100,
                "Processing complete!",
                dbc.Alert("File processed successfully!", color="success"),
                log,
                False  # Keep interval enabled briefly to show completion
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            return (
                no_update,
                no_update,
                progress if 'progress' in locals() else 0,
                f"Error: {str(e)}",
                dbc.Alert(error_msg, color="danger"),
                update_log(error_msg),
                True  # Disable interval on error
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
        [Input('dataset-selector', 'value')],
        [State('dataset-manifest-store', 'data')],
        prevent_initial_call=False
    )
    def update_bibliography_summary(selected_dataset_id, manifest):
        """Generate and display bibliography summary when a dataset is selected."""
        manifest = manifest or []
        entry = get_manifest_entry(manifest, selected_dataset_id)
        if not entry:
            return "Select or upload a dataset to see summary details.", {"display": "none"}

        try:
            file_path = upload_dir / entry['stored_name']
            return generate_bibliography_summary(file_path)
        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"Error generating summary: {exc}"
            return dbc.Alert(error_msg, color="danger"), {"display": "block"}
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
            # Process the file with deduplication
            logs = logger.log_info(f"Reading {filename}...")
            logger.set_progress(20)
            processor = BibtexProcessor()
            entries = processor.load_entries(str(file_path), deduplicate=True)
            dedup_stats = processor.get_deduplication_stats()
            logger.set_progress(40)
            
            if dedup_stats['duplicates_removed'] > 0:
                logs = logger.log_info(f"Found {dedup_stats['total_entries']} entries, deduplicated to {len(entries)} unique entries")
            else:
                logs = logger.log_info(f"Found {len(entries)} entries in the file")
            
            # Apply year filtering if specified
            if min_year or max_year:
                logger.set_progress(50)
                # Processor already has the entries loaded
                
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
         Input('dataset-selector', 'value')]
    )
    def update_search_button(query, data, dataset_id):
        """Enable search button when query is entered and data is available."""
        has_data = bool(data) or bool(dataset_id)
        
        if not has_data:
            return True, "Select or upload a dataset first to enable search"
        elif not query or not query.strip():
            return True, "Enter a search query (e.g., 'chronic fatigue syndrome')"
        else:
            # If we have a query and some form of data, enable the button
            return False, "Ready to search your uploaded bibliography"
    
    @app.callback(
        [Output('search-progress-card', 'style', allow_duplicate=True),
         Output('search-progress', 'value', allow_duplicate=True),
         Output('search-progress-text', 'children', allow_duplicate=True),
         Output('search-logs', 'children', allow_duplicate=True),
         Output('search-interval', 'disabled', allow_duplicate=True),
         Output('search-status', 'children', allow_duplicate=True),
         Output('search-results-container', 'children', allow_duplicate=True),
         Output('search-params-store', 'data', allow_duplicate=True),
        Output('download-search-csv-button', 'disabled', allow_duplicate=True),
        Output('download-staff-summary-button', 'disabled', allow_duplicate=True),
        Output('generate-staff-summaries', 'disabled', allow_duplicate=True),
        Output('staff-summary-status', 'children', allow_duplicate=True),
        Output('search-pagination-store', 'data', allow_duplicate=True),
        Output('search-summary-store', 'data', allow_duplicate=True),
        Output('results-visibility-store', 'data', allow_duplicate=True)],
        [Input('search-button', 'n_clicks')],
        [State('search-query', 'value'),
         State('search-methods', 'value'),
         State('semantic-threshold', 'value'),
         State('fuzzy-threshold', 'value'),
         State('max-results', 'value'),
         State('hybrid-model', 'value'),
         State('llm-threshold', 'value'),
         State('dataset-selector', 'value'),
         State('dataset-manifest-store', 'data')],
        prevent_initial_call=True
    )
    def initiate_search(n_clicks, query, methods, semantic_threshold, fuzzy_threshold,
                        max_results, hybrid_model, llm_threshold, dataset_id, manifest):
        """Prime the search progress UI and capture parameters for execution."""
        if not n_clicks or not query or not query.strip():
            raise PreventUpdate

        manifest = manifest or []
        trimmed_query = query.strip()

        search_progress.__init__()
        search_progress.update(0, f"Starting search for '{trimmed_query}'...", status='searching')
        state = search_progress.get_state()

        params = {
            "query": trimmed_query,
            "methods": methods or [],
            "semantic_threshold": semantic_threshold,
            "fuzzy_threshold": fuzzy_threshold,
            "max_results": max_results,
            "hybrid_model": hybrid_model,
            "llm_threshold": llm_threshold,
            "dataset_id": dataset_id,
            "manifest": manifest,
            "started_at": datetime.utcnow().isoformat()
        }

        loading_indicator = dcc.Loading(
            type="circle",
            children=html.Div("Preparing search...", className="text-muted mb-2")
        )

        return (
            {"display": "block"},
            state['progress'],
            f"Progress: {state['progress']}%",
            state['logs'],
            False,
            "Searching...",
            html.Div([loading_indicator]),
            params,
            True,
            True,
            True,
            "Generating search results…",
            {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0},
            None,
            {'show': False}
        )

    @app.callback(
        [Output('search-results-container', 'children', allow_duplicate=True),
         Output('search-status', 'children', allow_duplicate=True),
         Output('search-progress', 'value', allow_duplicate=True),
         Output('search-progress-text', 'children', allow_duplicate=True),
         Output('search-logs', 'children', allow_duplicate=True),
         Output('search-progress-card', 'style', allow_duplicate=True),
         Output('search-results-store', 'data', allow_duplicate=True),
         Output('search-interval', 'disabled', allow_duplicate=True),
         Output('staff-analysis-store', 'data', allow_duplicate=True),
         Output('staff-publication-map', 'data', allow_duplicate=True),
         Output('staff-analysis-container', 'children', allow_duplicate=True),
         Output('embedding-store', 'data', allow_duplicate=True),
         Output('search-params-store', 'data', allow_duplicate=True),
         Output('download-search-csv-button', 'disabled', allow_duplicate=True),
         Output('download-staff-summary-button', 'disabled', allow_duplicate=True),
         Output('generate-staff-summaries', 'disabled', allow_duplicate=True),
         Output('staff-summary-status', 'children', allow_duplicate=True),
         Output('search-pagination-store', 'data', allow_duplicate=True),
         Output('search-summary-store', 'data', allow_duplicate=True),
         Output('results-visibility-store', 'data', allow_duplicate=True)],
        [Input('search-params-store', 'data')],
        [State('data-store', 'data'),
         State('embedding-store', 'data')],
        prevent_initial_call=True
    )
    def perform_search(params, data, embedding_store):
        """Execute the semantic search using captured parameters."""
        if not params:
            raise PreventUpdate

        query = params.get('query')
        methods = params.get('methods', [])
        semantic_threshold = params.get('semantic_threshold')
        fuzzy_threshold = params.get('fuzzy_threshold')
        raw_display_limit = params.get('max_results')
        display_limit: Optional[int] = None
        if raw_display_limit is not None and str(raw_display_limit).strip():
            try:
                parsed_limit = int(float(raw_display_limit))
                if parsed_limit > 0:
                    display_limit = parsed_limit
            except (ValueError, TypeError):
                display_limit = None
        hybrid_model = params.get('hybrid_model')
        llm_threshold = params.get('llm_threshold')
        dataset_id = params.get('dataset_id')
        manifest = params.get('manifest', [])

        if not query:
            return (
                "",
                "Ready to search...",
                0,
                "",
                "",
                {"display": "none"},
                None,
                True,
                None,
                {},
                "",
                embedding_store,
                None,
                True,
                True,
                True,
                "Run a search to enable LLM summaries.",
                {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0},
                None,
                {'show': False}
            )

        logger = ProgressLogger(search_progress)
        progress_style = {"display": "block"}

        try:
            methods = methods or []
            logs = logger.log_info(f"Starting search for: '{query}'")
            logger.set_progress(5)

            entry = get_manifest_entry(manifest, dataset_id)

            if data:
                df = pd.read_json(data, orient='split').reset_index(drop=True)
                logs = logger.log_info(f"Loaded {len(df)} papers from data store")
            elif entry:
                dataset_name = entry.get('original_name', entry.get('stored_name', 'dataset'))
                logs = logger.log_info(f"No processed data in store, loading dataset: {dataset_name}")
                file_path = upload_dir / entry['stored_name']
                if not file_path.exists():
                    error_msg = f"Dataset file '{entry['stored_name']}' is missing on disk."
                    logs = logger.log_error(error_msg)
                    search_progress.update(0, status='error')
                    return (
                        dbc.Alert(error_msg, color="danger"),
                        "Error",
                        0,
                        "Error",
                        logs,
                        progress_style,
                        None,
                        True,
                        None,
                        {},
                        "",
                        embedding_store,
                        None,
                        True,
                        True,
                        True,
                        "Search failed before summaries could be generated.",
                        {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0},
                        None,
                        {'show': False}
                    )
                entries = process_bibtex_file(str(file_path))
                df = pd.DataFrame(entries).reset_index(drop=True)
                logs = logger.log_info(f"Loaded {len(df)} papers from dataset file")
            else:
                error_msg = "Select or upload a dataset before searching."
                logs = logger.log_error(error_msg)
                search_progress.update(0, status='error')
                return (
                    dbc.Alert(error_msg, color="danger"),
                    "Error",
                    0,
                    "Error",
                    logs,
                    progress_style,
                    None,
                    True,
                    None,
                    {},
                    "",
                    embedding_store,
                    None,
                    True,
                    True,
                    True,
                    "Select a dataset and run a search to enable summaries.",
                    {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0},
                    None,
                    {'show': False}
                )

            progress = 10
            logger.set_progress(progress)

            embedding_metadata = embedding_store if isinstance(embedding_store, dict) else {}
            embeddings = None

            if dataset_id:
                if embedding_metadata.get('dataset_id') == dataset_id and embedding_metadata.get('path'):
                    path = Path(embedding_metadata['path'])
                    if path.exists():
                        try:
                            data_npz = np.load(path, allow_pickle=False)
                            candidate_embeddings = data_npz['embeddings']
                            row_index = data_npz['row_index']
                            if candidate_embeddings.shape[0] == len(df) and np.array_equal(row_index, np.arange(len(df))):
                                embeddings = candidate_embeddings
                                logger.log_info('Loaded cached embeddings aligned with dataset.')
                            else:
                                logger.log_info('Cached embeddings mismatch dataset shape; recomputing.')
                        except Exception as exc:
                            logger.log_error(f"Failed to load cached embeddings: {exc}")
                if embeddings is None:
                    embeddings, embedding_metadata = ensure_embeddings_for_dataset(dataset_id, df, logger)
            else:
                embedding_metadata = embedding_metadata or {"status": "no_dataset"}

            try:
                if 'hybrid' in methods or 'llm_only' in methods:
                    searcher = HybridSemanticSearcher(llm_model=hybrid_model)
                    if 'llm_only' in methods:
                        logs = logger.log_info(f"Initialized LLM-only searcher ({hybrid_model})")
                    else:
                        logs = logger.log_info(f"Initialized hybrid searcher (embeddings + {hybrid_model})")
                else:
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key and 'semantic' in methods:
                        error_msg = "OpenAI API key required for semantic search. Please set OPENAI_API_KEY in your .env file."
                        logs = logger.log_error(error_msg)
                        search_progress.update(0, status='error')
                        return (
                            dbc.Alert(error_msg, color="danger"),
                            "Error",
                            0,
                            "Error",
                            logs,
                            progress_style,
                            None,
                            True,
                            None,
                            {},
                            "",
                            embedding_metadata,
                            None,
                            True,
                            True,
                            True,
                            "Add an OpenAI API key to enable semantic search and summaries.",
                            {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0},
                            None,
                            {'show': False}
                        )
                    searcher = SemanticSearcher(api_key=api_key) if api_key else SemanticSearcher()
                    logs = logger.log_info('Initialized standard semantic searcher')
            except Exception as exc:
                error_msg = f"Failed to initialize searcher: {exc}"
                logs = logger.log_error(error_msg)
                search_progress.update(0, status='error')
                return (
                    dbc.Alert(error_msg, color="danger"),
                    "Error",
                    0,
                    "Error",
                    logs,
                    progress_style,
                    None,
                    True,
                    None,
                    {},
                    "",
                    embedding_metadata,
                    None,
                    True,
                    True,
                    True,
                    "Search failed before summaries could be generated.",
                    {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0},
                    None,
                    {'show': False}
                )

            progress = 20
            logger.set_progress(progress)
            logs = logger.log_info(f"Search methods: {', '.join(methods)}")
            logs = logger.log_info(f"Thresholds - Semantic: {semantic_threshold:.2f}, Fuzzy: {fuzzy_threshold}%")

            base_methods = [m for m in methods if m in ('exact', 'fuzzy', 'semantic')]
            include_hybrid = 'hybrid' in methods
            include_llm_only = 'llm_only' in methods

            def ensure_score_fields(row_dict: Dict[str, Any]) -> Dict[str, Any]:
                defaults: Dict[str, Any] = {
                    'search_score': row_dict.get('search_score', 0.0),
                    'exact_score': row_dict.get('exact_score', 0.0),
                    'fuzzy_score': row_dict.get('fuzzy_score', 0.0),
                    'semantic_score': row_dict.get('semantic_score', 0.0),
                    'hybrid_score': row_dict.get('hybrid_score', 0.0),
                    'llm_only_score': row_dict.get('llm_only_score', 0.0),
                    'llm_relevance_score': row_dict.get('llm_relevance_score', 0.0),
                    'llm_confidence': row_dict.get('llm_confidence', 0.0)
                }
                for key, default in defaults.items():
                    if key not in row_dict:
                        row_dict[key] = default
                return row_dict

            results_map: Dict[int, Dict[str, Any]] = {}

            if base_methods:
                logger.log_info(f"Running layered search (exact → fuzzy → semantic) across {len(df)} papers")
                multi_results = searcher.multi_search(
                    query=query,
                    df=df,
                    methods=base_methods,
                    semantic_threshold=semantic_threshold,
                    fuzzy_threshold=fuzzy_threshold,
                    max_results=None,
                    logger=logger,
                    precomputed_embeddings=embeddings
                )
                if not multi_results.empty:
                    for _, row in multi_results.iterrows():
                        idx = int(row.get('original_index', row.name))
                        row_dict = ensure_score_fields(row.to_dict())
                        row_dict['original_index'] = idx
                        results_map[idx] = row_dict
                    logger.log_info(f"Collected {len(results_map)} candidates from lexical + embedding search")

            if include_hybrid:
                DEFAULT_EMBEDDING_CANDIDATES = 200
                candidate_limit = display_limit if display_limit else DEFAULT_EMBEDDING_CANDIDATES
                if candidate_limit < 50:
                    candidate_limit = 50
                logger.log_info("Running hybrid rerank (embeddings + LLM) on collected candidates")
                hybrid_results = searcher.hybrid_search(
                    query=query,
                    df=df,
                    threshold=semantic_threshold,
                    max_embedding_candidates=candidate_limit,
                    max_results=None,
                    logger=logger,
                    precomputed_embeddings=embeddings
                )
                analyzed_papers = getattr(searcher, '_last_analyzed_papers', [])
                llm_lookup = {p.get('original_index'): p for p in analyzed_papers if p.get('original_index') is not None}
                for idx, score in hybrid_results:
                    source_row = results_map.get(idx)
                    if source_row is None:
                        base_row = df.iloc[idx].to_dict()
                        base_row['original_index'] = idx
                        source_row = ensure_score_fields(base_row)
                    else:
                        source_row = ensure_score_fields(source_row)
                    source_row['hybrid_score'] = score
                    source_row['search_score'] = max(float(source_row.get('search_score', 0.0)), float(score))
                    llm_paper = llm_lookup.get(idx)
                    if llm_paper:
                        source_row['llm_relevance_score'] = llm_paper.get('llm_relevance_score', source_row.get('llm_relevance_score', 0.0))
                        source_row['llm_confidence'] = llm_paper.get('llm_confidence', source_row.get('llm_confidence', 0.0))
                        source_row['llm_reasoning'] = llm_paper.get('llm_reasoning', source_row.get('llm_reasoning', ''))
                        source_row['llm_key_concepts'] = llm_paper.get('llm_key_concepts', source_row.get('llm_key_concepts', []))
                    results_map[idx] = source_row
                logger.log_info(f"Hybrid rerank scored {len(hybrid_results)} papers; combined candidate pool: {len(results_map)}")

            if include_llm_only:
                logger.log_info("Running full LLM analysis (LLM Only mode)")
                llm_results = searcher.llm_only_search(
                    query=query,
                    df=df,
                    max_results=None,
                    relevance_threshold=llm_threshold,
                    logger=logger
                )
                analyzed_papers = getattr(searcher, '_last_analyzed_papers', [])
                llm_lookup = {p.get('original_index'): p for p in analyzed_papers if p.get('original_index') is not None}
                for idx, score in llm_results:
                    source_row = results_map.get(idx)
                    if source_row is None:
                        base_row = df.iloc[idx].to_dict()
                        base_row['original_index'] = idx
                        source_row = ensure_score_fields(base_row)
                    else:
                        source_row = ensure_score_fields(source_row)
                    source_row['llm_only_score'] = score
                    source_row['search_score'] = max(float(source_row.get('search_score', 0.0)), float(score))
                    llm_paper = llm_lookup.get(idx)
                    if llm_paper:
                        source_row['llm_relevance_score'] = llm_paper.get('llm_relevance_score', source_row.get('llm_relevance_score', 0.0))
                        source_row['llm_confidence'] = llm_paper.get('llm_confidence', source_row.get('llm_confidence', 0.0))
                        source_row['llm_reasoning'] = llm_paper.get('llm_reasoning', source_row.get('llm_reasoning', ''))
                        source_row['llm_key_concepts'] = llm_paper.get('llm_key_concepts', source_row.get('llm_key_concepts', []))
                    results_map[idx] = source_row
                logger.log_info(f"LLM-only scoring retained {len(llm_results)} papers; combined candidate pool: {len(results_map)}")

            if not results_map and not include_hybrid and not include_llm_only:
                # No methods selected that produced results
                results = pd.DataFrame()
            else:
                results = pd.DataFrame(results_map.values()) if results_map else pd.DataFrame()

            progress = 90
            logger.set_progress(progress)

            if results.empty:
                logs = logger.log_info('No results found with current settings')
                search_progress.update(100, 'No results found', status='complete')
                empty_pagination = {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0}
                return (
                    dbc.Alert(
                        f"No results found for '{query}' with the current settings. Try lowering the thresholds or using different search methods.",
                        color='warning'
                    ),
                    "",
                    100,
                    'Search complete - No results',
                    logs,
                    progress_style,
                    None,
                    True,
                    None,
                    {},
                    dbc.Card(
                        [
                            dbc.CardHeader("Staff Analysis"),
                            dbc.CardBody(html.Div('No staff contributors found for this query.', className='text-muted'))
                        ],
                        className="mb-3"
                    ),
                    embedding_metadata,
                    None,
                    True,
                    True,
                    True,
                    "No staff summaries available until results are found.",
                    empty_pagination,
                    None,
                    {'show': False}
                )

            logs = logger.log_info(f"Formatting {len(results)} results for display...")

            score_columns = [
                'search_score',
                'exact_score',
                'fuzzy_score',
                'semantic_score',
                'hybrid_score',
                'llm_only_score',
                'llm_relevance_score',
                'llm_confidence'
            ]
            for col in score_columns:
                if col in results.columns:
                    results[col] = results[col].fillna(0)
                else:
                    results[col] = 0

            if not results.empty:
                results['display_score'] = results[['hybrid_score', 'llm_only_score', 'search_score']].max(axis=1)
                results = results.sort_values(['display_score', 'search_score'], ascending=False)
                results = results.drop(columns=['display_score'])
                results = results.reset_index(drop=True)
                results['result_row_id'] = np.arange(len(results), dtype=int)
            else:
                results = results.assign(result_row_id=pd.Series(dtype=int))

            total_results = len(results)
            displayed_results = min(total_results, display_limit) if display_limit else total_results
            pagination_data = {
                'page': 1,
                'page_size': PAGE_SIZE,
                'total_results': displayed_results
            }

            progress = 100
            logger.set_progress(progress)
            search_progress.update(100, 'Search complete', status='complete')

            search_results_data = results.to_json(orient='split')

            staff_analyzer = StaffAnalyzer()
            staff_summary = staff_analyzer.generate_staff_summary(results)
            staff_publication_map = staff_analyzer.map_staff_to_publications(results)
            if not staff_summary.empty:
                if 'llm_summary' not in staff_summary.columns:
                    staff_summary['llm_summary'] = None
                if 'llm_focus_topics' not in staff_summary.columns:
                    staff_summary['llm_focus_topics'] = None
            staff_data = staff_summary.to_json(orient='split') if not staff_summary.empty else None
            if staff_summary.empty:
                staff_ui = dbc.Card(
                    [
                        dbc.CardHeader("Staff Analysis"),
                        dbc.CardBody(html.Div('No staff contributors found in these results.', className='text-muted'))
                    ],
                    className="mb-3"
                )
            else:
                staff_ui = dbc.Card(
                    [
                        dbc.CardHeader(f"Staff Analysis – Top 10 ({len(staff_summary)} total)"),
                        dbc.CardBody(staff_analyzer.render_staff_summary(staff_summary))
                    ],
                    className="mb-3"
                )
            staff_summary_status_text = (
                html.Span(
                    "Click “Generate LLM Staff Summaries” to highlight how each contributor relates to this search.",
                    className="text-muted small"
                )
                if not staff_summary.empty
                else "No staff contributors to summarize for this search."
            )
            generate_summaries_disabled = staff_summary.empty
            
            method_counts = {}
            if 'exact_score' in results.columns:
                method_counts['exact'] = int((results['exact_score'] > 0).sum())
            if 'fuzzy_score' in results.columns:
                method_counts['fuzzy'] = int((results['fuzzy_score'] > 0).sum())
            if 'semantic_score' in results.columns:
                method_counts['semantic'] = int((results['semantic_score'] > 0).sum())
            if 'hybrid_score' in results.columns:
                method_counts['hybrid'] = int((results['hybrid_score'] > 0).sum())
            if 'llm_only_score' in results.columns:
                method_counts['llm_only'] = int((results['llm_only_score'] > 0).sum())

            staff_total = int(len(staff_summary)) if not staff_summary.empty else 0
            avg_pubs = float(round(staff_summary['publication_count'].mean(), 2)) if staff_total else 0.0

            summary_data = {
                "total_results": int(total_results),
                "displayed_results": int(displayed_results),
                "method_counts": method_counts,
                "methods_selected": methods,
                "staff_total": staff_total,
                "avg_publications_per_staff": avg_pubs,
                "display_limit": display_limit if display_limit else None
            }
            download_results_disabled = total_results == 0
            download_staff_disabled = staff_summary.empty

            return (
                no_update,
                'Search complete',
                100,
                'Search complete!',
                logger.log_info('Search finished successfully'),
                progress_style,
                search_results_data,
                True,
                staff_data,
                staff_publication_map,
                staff_ui,
                embedding_metadata,
                None,
                download_results_disabled,
                download_staff_disabled,
                generate_summaries_disabled,
                staff_summary_status_text,
                pagination_data,
                summary_data,
                {'show': False}
            )

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"Search failed: {exc}"
            logs = logger.log_error(f"{error_msg}\n{tb}") if 'logger' in locals() else f"[ERROR] {error_msg}\n{tb}"
            search_progress.update(0, status='error')
            error_pagination = {'page': 1, 'page_size': PAGE_SIZE, 'total_results': 0}
            return (
                dbc.Alert([
                    html.Div(error_msg),
                    html.Pre(tb, className="mt-2 small text-muted")
                ], color='danger'),
                'Error occurred',
                0,
                'Error occurred',
                logs,
                progress_style,
                None,
                True,
                None,
                {},
                '',
                embedding_store if isinstance(embedding_store, dict) else {},
                None,
                True,
                True,
                True,
                "Search failed before summaries could be generated.",
                error_pagination,
                None,
                {'show': False}
            )

    @app.callback(
        Output('search-pagination-store', 'data', allow_duplicate=True),
        [Input('search-prev-page', 'n_clicks'),
         Input('search-next-page', 'n_clicks')],
        [State('search-pagination-store', 'data')],
        prevent_initial_call=True
    )
    def update_search_page(prev_clicks, next_clicks, pagination):
        """Handle pagination controls for search results."""
        if pagination is None:
            raise PreventUpdate

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        page = pagination.get('page', 1) or 1
        page_size = pagination.get('page_size', PAGE_SIZE) or PAGE_SIZE
        total_results = pagination.get('total_results', 0) or 0
        total_pages = max(1, math.ceil(total_results / page_size)) if page_size else 1

        if triggered == 'search-prev-page' and page > 1:
            pagination['page'] = page - 1
        elif triggered == 'search-next-page' and page < total_pages:
            pagination['page'] = page + 1
        else:
            raise PreventUpdate

        return pagination

    @app.callback(
        [Output('staff-analysis-store', 'data', allow_duplicate=True),
         Output('staff-analysis-container', 'children', allow_duplicate=True),
         Output('staff-summary-status', 'children', allow_duplicate=True),
         Output('generate-staff-summaries', 'disabled', allow_duplicate=True)],
        Input('generate-staff-summaries', 'n_clicks'),
        [State('staff-analysis-store', 'data'),
         State('staff-publication-map', 'data'),
         State('search-results-store', 'data'),
         State('search-query', 'value'),
         State('dataset-selector', 'value'),
         State('hybrid-model', 'value')],
        prevent_initial_call=True
    )
    def generate_staff_llm_summaries(n_clicks, staff_data_json, publication_map, results_json, query, dataset_id, llm_model):
        """Generate concise LLM summaries for the top staff contributors."""
        if not n_clicks or not staff_data_json or not results_json:
            raise PreventUpdate

        staff_df = pd.read_json(staff_data_json, orient='split')
        if staff_df.empty:
            status = "No staff contributors to summarize for this search."
            return staff_data_json, no_update, status, True

        if 'llm_summary' not in staff_df.columns:
            staff_df['llm_summary'] = None
        else:
            staff_df['llm_summary'] = staff_df['llm_summary'].astype('object')
        if 'llm_focus_topics' not in staff_df.columns:
            staff_df['llm_focus_topics'] = None
        else:
            staff_df['llm_focus_topics'] = staff_df['llm_focus_topics'].astype('object')

        publication_map = publication_map or {}
        results_df = pd.read_json(results_json, orient='split')
        summarizer = StaffLLMSummarizer(model=llm_model or "gpt-4o-mini")
        processed = 0
        for _, row in staff_df.head(10).iterrows():
            staff_id = str(row['staff_id'])
            result_ids = publication_map.get(staff_id) or []
            if not result_ids:
                continue
            staff_matches = results_df[results_df['result_row_id'].isin(result_ids)]
            if staff_matches.empty:
                continue
            summary_payload = summarizer.summarize_staff(
                query=query or "",
                staff_id=staff_id,
                staff_row=row.to_dict(),
                papers=staff_matches,
                dataset_id=dataset_id or "uploaded"
            )
            staff_df.loc[staff_df['staff_id'] == row['staff_id'], 'llm_summary'] = summary_payload.get('summary')
            focus_topics = summary_payload.get('focus_topics', [])
            if isinstance(focus_topics, list):
                focus_topics = json.dumps(focus_topics)
            staff_df.loc[staff_df['staff_id'] == row['staff_id'], 'llm_focus_topics'] = focus_topics
            processed += 1

        staff_analyzer = StaffAnalyzer()
        staff_ui = (
            dbc.Card(
                [
                    dbc.CardHeader("Staff Analysis"),
                    dbc.CardBody(html.Div('No staff contributors found in these results.', className='text-muted'))
                ],
                className="mb-3"
            )
            if staff_df.empty
            else dbc.Card(
                [
                    dbc.CardHeader(f"Staff Analysis – Top 10 ({len(staff_df)} total)"),
                    dbc.CardBody(staff_analyzer.render_staff_summary(staff_df))
                ],
                className="mb-3"
            )
        )

        updated_store = staff_df.to_json(orient='split')
        status = (
            html.Span(
                f"Generated summaries for {processed} staff members.",
                className="text-success small"
            )
            if processed
            else html.Span(
                "Unable to generate summaries with the current data.",
                className="text-warning small"
            )
        )
        return (
            updated_store,
            staff_ui,
            status,
            False
        )

    @app.callback(
        Output('results-visibility-store', 'data', allow_duplicate=True),
        Input('toggle-results-btn', 'n_clicks'),
        State('results-visibility-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_results_visibility(n_clicks, visibility):
        if n_clicks is None:
            raise PreventUpdate
        current = visibility or {'show': False}
        return {'show': not current.get('show', False)}

    @app.callback(
        [Output('toggle-results-btn', 'children'),
         Output('toggle-results-btn', 'color')],
        Input('results-visibility-store', 'data'),
        prevent_initial_call=False
    )
    def update_toggle_button(visibility):
        show = visibility.get('show', False) if isinstance(visibility, dict) else False
        return (
            "Hide Results" if show else "Show Results",
            "danger" if show else "primary"
        )

    @app.callback(
        [Output('search-results-container', 'children', allow_duplicate=True),
         Output('pagination-info', 'children'),
         Output('search-prev-page', 'disabled', allow_duplicate=True),
         Output('search-next-page', 'disabled', allow_duplicate=True)],
        [Input('search-pagination-store', 'data'),
         Input('results-visibility-store', 'data')],
        [State('search-results-store', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def render_paginated_results(pagination, visibility, results_json):
        """Render the current page of results."""
        visible = visibility.get('show', False) if isinstance(visibility, dict) else False
        if not visible:
            return (
                html.Div("Search results hidden. Click “Show Results” to view the list.", className="text-muted"),
                "",
                True,
                True
            )

        if not results_json:
            return (
                no_update,
                "",
                True,
                True
            )

        df = pd.read_json(results_json, orient='split')
        if pagination is None:
            pagination = {'page': 1, 'page_size': PAGE_SIZE, 'total_results': len(df)}

        page = int(pagination.get('page', 1) or 1)
        page_size = int(pagination.get('page_size', PAGE_SIZE) or PAGE_SIZE)
        total_value = pagination.get('total_results', len(df))
        total_results = len(df) if total_value is None else int(total_value)

        if total_results == 0 or df.empty:
            info_text = "No results to display."
            return (
                html.Div("No search results available. Adjust your query and try again.", className="text-muted"),
                info_text,
                True,
                True
            )

        total_pages = max(1, math.ceil(total_results / page_size)) if page_size else 1
        page = max(1, min(page, total_pages))

        limited_df = df.iloc[:total_results] if total_results < len(df) else df
        cards = create_result_cards(limited_df, page=page, page_size=page_size)
        info_text = format_pagination_text(page, page_size, total_results)
        prev_disabled = page <= 1
        next_disabled = page >= total_pages

        header = dbc.Alert(
            f"Found {total_results} result{'s' if total_results != 1 else ''}",
            color='success',
            className='mb-3'
        )

        return (
            html.Div([header, *cards]),
            info_text,
            prev_disabled,
            next_disabled
        )
    
    @app.callback(
        Output('search-summary-card', 'children'),
        Input('search-summary-store', 'data'),
        prevent_initial_call=False
    )
    def render_summary_card(summary):
        """Render summary metrics for the current search."""
        if not summary:
            return dbc.Card(
                [
                    dbc.CardHeader("Results Summary"),
                    dbc.CardBody(html.Div("Run a search to see summary metrics.", className="text-muted"))
                ],
                className="mb-3"
            )

        total = summary.get('total_results', 0)
        method_counts = summary.get('method_counts', {})
        methods_selected = summary.get('methods_selected', [])
        staff_total = summary.get('staff_total', 0)
        avg_pubs = summary.get('avg_publications_per_staff', 0.0)
        displayed_results = summary.get('displayed_results', total)
        display_limit = summary.get('display_limit')

        method_items = []
        for method in ['exact', 'fuzzy', 'semantic', 'hybrid', 'llm_only']:
            if method in method_counts:
                label = method.replace('_', ' ').title()
                method_items.append(html.Li(f"{label}: {method_counts[method]}"))

        display_note = None
        if total:
            if display_limit and display_limit < total:
                display_note = html.P(
                    f"Showing the first {displayed_results} on screen. Download includes all {total} deduplicated matches.",
                    className="mb-2 text-muted fst-italic"
                )
            else:
                display_note = html.P(
                    "Downloads include every deduplicated match returned by the search.",
                    className="mb-2 text-muted fst-italic"
                )

        body_children = [
            html.P(f"Total deduplicated matches: {total}", className="mb-2 fw-semibold")
        ]
        if display_note:
            body_children.append(display_note)
        body_children.extend([
            html.P(f"Methods selected: {', '.join(methods_selected) if methods_selected else 'None'}", className="mb-2"),
            html.P([
                html.Strong("Staff contributors: "),
                f"{staff_total}",
                html.Span(f" · Avg publications per staff: {avg_pubs:.2f}", className="ms-1 text-muted")
            ], className="mb-3"),
            html.Ul(method_items, className="mb-0") if method_items else html.Div(
                "No method-specific matches counted.", className="text-muted"
            )
        ])

        return dbc.Card(
            [
                dbc.CardHeader("Results Summary"),
                dbc.CardBody(body_children)
            ],
            className="mb-3"
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
            results_df = pd.read_json(search_results_data, orient='split')
            if results_df.empty:
                raise PreventUpdate

            desired_columns = ['year', 'title', 'abstract', 'journal', 'author']
            for col in desired_columns:
                if col not in results_df.columns:
                    results_df[col] = ""

            export_df = results_df[desired_columns].copy()
            export_df = export_df.drop_duplicates(subset=['title', 'year']).reset_index(drop=True)

            safe_query = (query or "search")
            clean_query = "".join(c for c in safe_query if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_query = clean_query.replace(' ', '_')[:30]

            filename = f"{clean_query}_results_{len(export_df)}papers.csv"
            return dcc.send_data_frame(export_df.to_csv, filename=filename, index=False)

        except Exception:
            raise PreventUpdate

    @app.callback(
        Output('download-staff-analysis', 'data'),
        [Input('download-staff-summary-button', 'n_clicks')],
        [State('staff-analysis-store', 'data'),
         State('search-query', 'value')],
        prevent_initial_call=True
    )
    def download_staff_summary_csv(n_clicks, staff_data, query):
        """Download staff summary as CSV file."""
        if not n_clicks or not staff_data:
            raise PreventUpdate
        
        staff_df = pd.read_json(staff_data, orient='split')
        if staff_df.empty:
            raise PreventUpdate
        
        clean_query = "".join(c for c in (query or "search") if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_query = clean_query.replace(' ', '_')[:30]
        filename = f"staff_summary_{clean_query}_{len(staff_df)}staff.csv"
        
        return dcc.send_data_frame(staff_df.to_csv, filename=filename, index=False)
    
    @app.callback(
        [Output('staff-pubs-modal', 'is_open', allow_duplicate=True),
         Output('staff-pubs-modal-title', 'children', allow_duplicate=True),
         Output('staff-pubs-content', 'children', allow_duplicate=True)],
        Input({'type': 'staff-pub-btn', 'index': ALL}, 'n_clicks'),
        State('search-results-store', 'data'),
        State('staff-publication-map', 'data'),
        prevent_initial_call=True
    )
    def display_staff_publications(n_clicks_list, results_json, publication_map):
        """Open modal showing publications for a specific staff member."""
        if not n_clicks_list or all((clk or 0) == 0 for clk in n_clicks_list):
            raise PreventUpdate
        
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_prop = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            triggered_dict = json.loads(trigger_prop)
            staff_id = str(triggered_dict.get('index'))
        except (json.JSONDecodeError, TypeError):
            raise PreventUpdate
        
        if not staff_id:
            raise PreventUpdate
        
        if not results_json:
            content = html.Div("No search results available. Run a search first.", className="text-muted")
            return True, "Staff Publications", content

        df = pd.read_json(results_json, orient='split')

        staff_results = pd.DataFrame()
        if publication_map and isinstance(publication_map, dict):
            candidate_ids = publication_map.get(staff_id)
            if candidate_ids and 'result_row_id' in df.columns:
                staff_results = df[df['result_row_id'].isin(candidate_ids)].reset_index(drop=True)

        if staff_results.empty:
            def staff_matches(row):
                candidates = set()
                if 'all_staff_ids' in row and pd.notna(row['all_staff_ids']):
                    value = row['all_staff_ids']
                    if isinstance(value, list):
                        candidates.update(str(v).strip() for v in value if str(v).strip())
                    else:
                        candidates.update(str(v).strip() for v in str(value).replace(';', ',').split(',') if str(v).strip())
                if 'staff_id' in row and pd.notna(row['staff_id']):
                    candidates.add(str(row['staff_id']).strip())
                return staff_id in candidates

            staff_results = df[df.apply(staff_matches, axis=1)].reset_index(drop=True)
        if staff_results.empty:
            content = html.Div(
                "No publications found for this staff member in the current results.",
                className="text-muted"
            )
            title = f"{staff_id} – no publications found"
            return True, title, content

        cards = create_result_cards(staff_results, page=1, page_size=max(1, len(staff_results)))
        title = f"{staff_id} – {len(staff_results)} publication{'s' if len(staff_results) != 1 else ''}"
        content = html.Div(cards, className="staff-publications")

        return True, title, content

    @app.callback(
        Output('staff-pubs-modal', 'is_open', allow_duplicate=True),
        Input('staff-pubs-close', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_staff_modal(close_clicks):
        if close_clicks:
            return False
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
         Output('process-interval', 'disabled', allow_duplicate=True)],
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
    
    # Note: Interval enabling/disabling is handled by the main processing callbacks above


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
