# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bibtex Analyzer GPT is a tool for analyzing and visualizing bibliographies using AI-powered tagging. It supports both BibTeX (.bib) and CSV formats, provides year filtering, and generates interactive visualizations through OpenAI GPT models.

## Core Architecture

### Main Components

- **CLI Interface** (`__main__.py`): Entry point providing `analyze`, `visualize`, and `dashboard` commands
- **BibtexProcessor** (`bibtex_processor.py`): Handles file format detection, parsing (BibTeX/CSV), and year filtering with smart year extraction
- **TagGenerator** (`tag_generator.py`): Integrates with OpenAI API for AI-powered tag generation from abstracts
- **Dashboard** (`dashboard.py`): Dash web application for interactive analysis with file upload, real-time processing, and visualization controls
- **Visualization** (`visualization/`): Multiple visualization backends (matplotlib, plotly) for word clouds and network graphs

### Data Flow

1. **Input Processing**: Multi-format files (BibTeX/CSV) → `BibtexProcessor` → normalized entry dictionaries
2. **Filtering**: Year-based filtering using intelligent year extraction (handles formats like "2023", "c2023", "2023-01-01")
3. **AI Analysis**: Abstracts → `TagGenerator` → OpenAI API → topic tags
4. **Visualization**: Tagged data → various visualization modules → interactive/static outputs

### Key Design Patterns

- **Format Agnostic**: Core processing works with normalized dictionary entries regardless of input format
- **Modular Visualization**: Separate modules for different visualization backends (mpl, plotly, interactive)
- **Interactive Tag Generation**: CLI supports iterative tag refinement with user feedback
- **State Management**: Dashboard uses Dash callbacks for reactive UI updates

## Development Commands

### Environment Setup
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env  # Add OPENAI_API_KEY
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bibtex_processor.py

# Run single test with verbose output
pytest tests/test_tag_generator.py::test_generate_tags -v

# Run with coverage
pytest --cov=bibtex_analyzer --cov-report=html
```

### Code Quality
```bash
# Format code
black bibtex_analyzer/ tests/

# Sort imports
isort bibtex_analyzer/ tests/

# Lint
flake8 bibtex_analyzer/ tests/

# Type checking
mypy bibtex_analyzer/
```

### Running the Application
```bash
# CLI commands
python -m bibtex_analyzer analyze input.bib --output results.csv --min-year 2020
python -m bibtex_analyzer dashboard --port 8050 --debug
python -m bibtex_analyzer visualize results.csv --interactive

# Direct module execution for testing
python -c "from bibtex_analyzer import process_bibtex_file; print(process_bibtex_file('examples/sample_references.bib'))"
```

## Important Implementation Details

### File Format Support
- **BibTeX**: Uses `bibtexparser` library, handles standard BibTeX fields
- **CSV**: Automatic format detection, requires `title`, `author`, `abstract` columns, auto-generates IDs if missing
- **Year Filtering**: Smart extraction supports various formats, validates years (1000-2100), filters before tag generation

### Dashboard Architecture
- **Dash App**: Multi-page layout with Bootstrap components
- **File Upload**: Base64 encoding/decoding with temporary file storage
- **Real-time Logging**: Custom logger class for process feedback
- **State Management**: Uses `dcc.Store` components for data persistence between callbacks
- **Year Controls**: Min/Max year inputs integrated into processing workflow

### Tag Generation
- **OpenAI Integration**: Supports multiple models (GPT-3.5, GPT-4)
- **Interactive Mode**: CLI allows iterative refinement with user feedback
- **Batch Processing**: Handles large bibliographies with configurable sample sizes
- **Error Handling**: Graceful fallbacks for API failures

### Testing Strategy
- **Fixtures**: Shared test data in `conftest.py` with sample entries and mock responses
- **Integration Tests**: End-to-end testing of CLI commands and file processing
- **Mock Objects**: OpenAI API responses mocked for reliable testing
- **Temporary Files**: Uses pytest's `tmp_path` for file I/O testing

## Related Projects

This tool is designed to work with [Publication Enricher](https://github.com/ChrisOldmeadow/publication-enricher) for data quality improvement. The recommended workflow is:
1. Clean/enrich incomplete bibliographies with Publication Enricher
2. Analyze enriched data with Bibtex Analyzer GPT
3. Export results with complete metadata

## Configuration

Environment variables are loaded from `.env` file:
- `OPENAI_API_KEY`: Required for tag generation
- `DEFAULT_MODEL`: Default OpenAI model (gpt-3.5-turbo)
- `LOG_LEVEL`: Logging verbosity
- `PORT`: Dashboard port (default: 8050)