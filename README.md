# Bibtex Analyzer GPT

A powerful tool for analyzing and visualizing BibTeX and CSV bibliographies using AI-powered tagging and interactive visualizations. This tool helps researchers, academics, and anyone working with academic papers to better understand and explore their reference collections.

> **💡 Tip**: For incomplete bibliographies missing abstracts or metadata, first use [Publication Enricher](https://github.com/ChrisOldmeadow/publication-enricher) to clean and enrich your data, then analyze with Bibtex Analyzer GPT for optimal results.

## Features

- **Multi-format Processing**: Extract and process bibliography entries from `.bib` (BibTeX) and `.csv` files
- **AI-Powered Tagging**: Generate topic tags using OpenAI's GPT models
- **Interactive Visualizations**:
  - Optional word clouds (static PNG and/or interactive HTML)
  - Clickable word clouds that link to relevant papers
  - Tag frequency analysis
  - Interactive tag networks
- **Semantic Search**: Advanced topic-based search using AI embeddings:
  - Exact match (case-insensitive)
  - Fuzzy matching for typos and variations
  - Semantic similarity using OpenAI embeddings
  - Multi-level search combining all methods
- **Year Filtering**: Filter entries by publication year range
- **Export Options**: Save results in multiple formats (CSV, HTML, PNG, PDF)
- **Interactive Dashboard**: Web-based interface for exploring your bibliography with:
  - File upload and processing
  - Real-time word cloud generation
  - Tag frequency analysis
  - Data table with filtering and sorting
  - Semantic search interface for topic discovery
  - Export functionality
  - Customizable visualization options
- **Staff Insights**: Automatically deduplicates matched staff, ranks them via an impact score, surfaces the top 10 in the dashboard, and optionally generates GPT-powered blurbs explaining how each researcher relates to the active search term.

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for tag generation)

### Installation Methods

#### 1. Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install bibtex-analyzer

# Or install directly from GitHub
pip install git+https://github.com/ChrisOldmeadow/bibtex-analyzer-gpt.git
```

#### 2. From Source

```bash
# Clone the repository
git clone https://github.com/ChrisOldmeadow/bibtex-analyzer-gpt.git
cd bibtex-analyzer-gpt

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Configuration

1. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   DEFAULT_MODEL=gpt-3.5-turbo
   ```

## Usage

### Command Line Interface

```bash
# Analyze a BibTeX or CSV file and generate tags with optional word cloud
bibtex-analyzer analyze papers.bib --output results.csv --tag-samples 50 --wordcloud
# Or with a CSV file
bibtex-analyzer analyze bibliography.csv --output results.csv --tag-samples 50 --wordcloud

# Word cloud options:
# --wordcloud         # Default: generate PNG word cloud
# --wordcloud png     # Generate PNG word cloud
# --wordcloud html    # Generate interactive HTML word cloud
# --wordcloud both    # Generate both PNG and HTML word clouds
# (no --wordcloud)    # Skip word cloud generation

# Launch the interactive dashboard
bibtex-analyzer dashboard
# Optional arguments:
# --port 8050        # Port to run the dashboard on (default: 8050)
# --debug            # Run in debug mode

# Generate visualizations from tagged data
bibtex-analyzer visualize results.csv

# Search for papers by topic using semantic similarity
bibtex-analyzer search "chronic fatigue syndrome" results.csv --output search_results.csv
bibtex-analyzer search "machine learning" papers.csv --methods semantic --semantic-threshold 0.8

# Analyze a BibTeX or CSV file and generate tags
# Using default settings (30 samples for tag generation, tag all entries)
bibtex-analyzer analyze input.bib --output tagged_papers.csv

# Filter by publication year
bibtex-analyzer analyze input.bib --output recent_papers.csv --min-year 2020 --max-year 2023

# Customize tag generation and processing with year filtering
bibtex-analyzer analyze input.bib \
    --output tagged_papers.csv \
    --tag-samples 50 \
    --subset-size 200 \
    --min-year 2020 \
    --max-year 2023

# Use a subset of entries for faster processing
bibtex-analyzer analyze large_library.bib \
    --output sample_tags.csv \
    --tag-samples 30 \
    --subset-size 100

# Generate visualizations from tagged data
bibtex-analyzer visualize tagged_papers.csv --interactive

# Launch the interactive dashboard
bibtex-analyzer dashboard
```

### Python API

```python
from bibtex_analyzer import process_bibtex_file, TagGenerator, SemanticSearcher, search_papers

# Process a BibTeX or CSV file and get tagged entries
entries = process_bibtex_file("input.bib", output_file="output/tagged.csv")
# Or with CSV
entries = process_bibtex_file("bibliography.csv", output_file="output/tagged.csv")

# Generate tags for a list of entries
tag_generator = TagGenerator(api_key="your_api_key")
tags = tag_generator.generate_tags_for_abstracts(entries)

# Assign tags to entries
tagged_entries = tag_generator.assign_tags_to_abstracts(entries, tags)

# Search for papers by topic
results = search_papers(
    query="chronic fatigue syndrome",
    input_file="tagged_papers.csv",
    methods=["exact", "fuzzy", "semantic"],
    semantic_threshold=0.7
)

# Advanced semantic search
searcher = SemanticSearcher(api_key="your_api_key")
import pandas as pd
df = pd.read_csv("papers.csv")

# Multi-level search
search_results = searcher.multi_search(
    query="machine learning in healthcare",
    df=df,
    methods=["semantic"],
    semantic_threshold=0.75,
    max_results=20
)
```

### Command Line Options

```
usage: bibtex-analyzer [-h] {analyze,visualize,search,dashboard} ...

Bibtex Analyzer - Analyze and visualize BibTeX bibliographies

positional arguments:
  {analyze,visualize,search,dashboard}
    analyze             Analyze a BibTeX or CSV file and generate tags
    visualize           Generate visualizations from tagged data
    search              Search for papers by topic using semantic similarity
    dashboard           Launch interactive web dashboard

options:
  -h, --help            show this help message and exit
```

#### Analyze Command

```
usage: bibtex-analyzer analyze [-h] [--output OUTPUT] [--tag-samples TAG_SAMPLES]
                              [--subset-size SUBSET_SIZE] [--min-year MIN_YEAR]
                              [--max-year MAX_YEAR] [--model MODEL]
                              [--methods-model METHODS_MODEL]
                              input

positional arguments:
  input                 Input BibTeX or CSV file

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output CSV file (default: tagged_abstracts.csv)
  --tag-samples TAG_SAMPLES
                        Number of samples to use for tag generation (default: 30)
  --subset-size SUBSET_SIZE
                        Number of random entries to tag (0 for all) (default: 100)
  --min-year MIN_YEAR   Filter entries to include only those from this year onwards
  --max-year MAX_YEAR   Filter entries to include only those up to this year
  --model MODEL         Model to use for tag generation (default: gpt-3.5-turbo)
  --methods-model METHODS_MODEL
                        Model to use for statistical methods (default: gpt-4)
```

#### Search Command

```
usage: bibtex-analyzer search [-h] [-o OUTPUT]
                              [--methods {exact,fuzzy,semantic} [{exact,fuzzy,semantic} ...]]
                              [--semantic-threshold SEMANTIC_THRESHOLD]
                              [--fuzzy-threshold FUZZY_THRESHOLD]
                              [--max-results MAX_RESULTS]
                              query input

positional arguments:
  query                 Search query (e.g., 'chronic fatigue syndrome')
  input                 Input CSV file with paper data

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output CSV file for search results
  --methods {exact,fuzzy,semantic} [{exact,fuzzy,semantic} ...]
                        Search methods to use (default: all methods)
  --semantic-threshold SEMANTIC_THRESHOLD
                        Minimum similarity score for semantic search (0-1, default: 0.7)
  --fuzzy-threshold FUZZY_THRESHOLD
                        Minimum similarity score for fuzzy search (0-100, default: 80)
  --max-results MAX_RESULTS
                        Maximum number of results to return (default: 50)
```

### Setting Up

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

2. Run tests:
   ```bash
   pytest
   ```

## Theme Analysis CLI

The separate `theme_analysis` module runs hybrid semantic searches across strategic themes with SCImago-style scoring and staff rollups. Launch it with:

```bash
python scripts/run_theme_search.py \
  --dataset data/institutional_publications_enriched.csv \
  --themes themes.yaml \
  --output results/theme_analysis_2025/ \
  [--baselines data/baselines.json] \
  [--max-candidates 150] \
  [--candidate-prompt] \
  [--semantic-only] \
  [--ignore-max-limits] \
  [--semantic-threshold 0.6]
```

Key flags:

- `--max-candidates`: overrides each theme’s `max_candidates` value and caps how many papers per theme are sent to GPT for reranking (defaults to 100 unless set in YAML).
- `--candidate-prompt`: when the semantic filter finds more matches than the cap, the CLI pauses so you can keep the limit, send all candidates, or enter a custom number.
- `--semantic-only`: skips the GPT rerank entirely and relies on embedding similarity (fast + no API cost). You can also set `semantic_only: true` inside individual themes for mixed-mode runs.
- `--ignore-max-limits`: ignore any `max_candidates`/`max_results` values defined in `themes.yaml` so hybrid/semantic runs process every match that clears the thresholds (unless you explicitly pass `--max-candidates`).
- `--semantic-threshold`: override every theme’s `semantic_threshold` (0-1) with a single global value.

Per-theme overrides (`max_candidates`, `semantic_only`, `prompt_on_overflow`) can be defined in `themes.yaml` to customize behavior per narrative. See `theme_analysis/docs/README_THEME_ANALYSIS.md` for the full workflow, scoring methodology, and output details.

### Project Structure

- `bibtex_analyzer/`: Main package
  - `bibtex_processor.py`: BibTeX parsing and processing
  - `tag_generator.py`: AI-powered tag generation
  - `visualization/`: Visualization modules
  - `utils/`: Utility functions

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## Dashboard Features

The interactive dashboard provides a user-friendly web interface for exploring your bibliography with a clean, organized layout designed for efficient workflows.

## Dashboard Layout

The dashboard features a **two-panel layout** with clearly separated functionality:

### Left Panel: 📁 Import & Filter Bibliography
**Data Management Hub** - Everything you need to import and prepare your data:

- **File Upload**: Drag and drop your bibliography file (.bib or .csv) or click to browse
- **Year Filtering**: 
  - Min Year: Filter entries from this year onwards
  - Max Year: Filter entries up to this year
- **Apply Filters**: Process your uploaded file with the selected year range
- **Processing Logs**: Real-time progress monitoring with detailed logging
- **Summary Information**: View statistics about your imported/filtered bibliography

### Right Panel: Analysis & Search Tools
**Two specialized tabs** for different analysis workflows:

## Tab 1: 🔍 Interactive Search
**Powerful semantic search independent of tag generation** - works immediately after uploading your bibliography:

- **Natural Language Queries**: Search using plain English descriptions like "chronic fatigue syndrome" or "machine learning in healthcare"
- **Multiple Search Methods**:
  - **Exact Match**: Finds papers containing your exact keywords
  - **Fuzzy Match**: Handles typos and variations in terminology
  - **Semantic Match**: AI embeddings for topic similarity (fast)
  - **🚀 Hybrid AI**: Combines embeddings + LLM analysis for best balance
  - **🤖 LLM Only**: Premium quality - GPT analyzes ALL papers (expensive but thorough)
- **Layered Pipeline**: Every search builds the result set in stages—exact → fuzzy → semantic populate the candidate pool, hybrid optionally reranks those hits with GPT scoring, and LLM-only can score the entire library when you need exhaustive review.
- **Advanced Controls**:
  - AI model selection (GPT-4o Mini, GPT-3.5 Turbo, GPT-4o, GPT-4 Turbo)
  - Semantic similarity thresholds (0.1-1.0)
  - Fuzzy matching thresholds (50-100%)
  - LLM relevance thresholds (0-10)
  - Maximum results limit (optional UI-only cap; CSV downloads always contain every deduplicated match)
- **Search Results**: View results with relevance scores, search method breakdown, and paper details
- **CSV Export**: Download search results with all metadata including publication_id and relevance scores

## Tab 2: 🏷️ AI Tags & Word Clouds
**Complete tag generation and visualization workflow**:

### Tag Generation Settings
- **Tag Sample Size**: Number of entries to use for generating tag vocabulary (1-∞)
- **Max Entries to Tag**: Limit entries to tag for faster processing (0 for all)
- **Model Selection**: Choose between GPT-3.5 Turbo or GPT-4 for tag generation
- **Process & Generate Tags**: Apply filters AND generate AI tags (required for visualizations)

### Word Cloud Customization (Collapsible)
- **Maximum Words**: Control word cloud density (10-200 words)
- **Color Schemes**: Viridis, Plasma, Inferno, Magma, Cividis, Rainbow
- **Background Colors**: White, Black, Light Gray, Dark
- **Word Cloud Type**: Static (PNG) or Interactive (HTML)
- **Update Word Cloud**: Regenerate with new settings

### 📊 Results & Visualizations (Nested Tabs)
*Only available after tag generation - displays results of the tagging process:*

#### Word Cloud Tab
- **Interactive/Static Word Clouds**: Beautiful visualizations of your paper topics
- **Clickable Words**: Click on any word to see related papers
- **Download Options**: Save as PNG or interactive HTML
- **Selected Word Display**: View papers associated with clicked words

#### Tag Frequencies Tab
- **Bar Chart Visualization**: Shows most common tags across your bibliography
- **Interactive Elements**: Hover for exact counts, click to filter papers
- **Download**: Export frequency data

#### Data Table Tab
- **Complete Bibliography View**: All entries with comprehensive metadata:
  - Publication ID, Year, Title, Authors, Journal, DOI, Abstract
  - Volume, Issue, Pages, Keywords, Generated Tags
- **Advanced Filtering**: 
  - Filter by generated tags using searchable multi-select dropdown
  - View filtered subsets of your bibliography
- **Download Options**:
  - **Download All Entries**: Complete bibliography as BibTeX
  - **Download Filtered Entries**: Only papers matching selected tag filters
- **Rich Metadata**: All original fields plus AI-generated tags and keywords

## Search vs. Tag Generation Workflows

### 🔍 **Search Workflow** (Independent & Fast)
1. Upload bibliography → 2. Use Search tab → 3. Get instant results
- **When to use**: Quick topic discovery, finding specific papers, exploring content
- **Speed**: Very fast - works immediately after upload
- **Output**: Ranked search results with relevance scores

### 🏷️ **Tag Generation Workflow** (Comprehensive & Slower)  
1. Upload bibliography → 2. Apply filters → 3. Generate tags → 4. Explore visualizations
- **When to use**: Deep analysis, topic modeling, creating word clouds, data exploration
- **Speed**: Slower - requires AI processing time
- **Output**: Tagged bibliography + visualizations + enhanced data table

### Layered Search & Staff Insights

- **Stepwise ranking**: Exact, fuzzy, and semantic passes build a deduplicated candidate pool. Hybrid mode reranks those matches with GPT relevance/confidence, and LLM-only mode can analyze every paper when you need exhaustive scoring.
- **Transparent limits**: The optional *Max Results* field only caps how many cards are shown per page; CSV exports always include the full deduplicated result set plus all method scores.
- **Summary metrics**: Each run renders a “Results Summary” card with total matches, method breakdown, selected search modes, and staff stats (count + average publications per staff).
- **Staff leaderboard**: The dashboard ranks staff by an impact score (log-scaled publication count + relevance + quality + citations) and highlights the top 10 with quick links. The download button exports every contributor, and the modal buttons open only the publications that matched your current search.
- **LLM narratives**: After reviewing the leaderboard, click “Generate LLM Staff Summaries” to have GPT craft short overviews for the top 10 researchers. The blurbs incorporate each staff member’s lead/senior authorship share, international collaboration rate, and clinical-trial involvement; they appear under each card and land in the staff summary download alongside the focus keywords.

## Key Features Summary

- **Dual Workflow Support**: Search immediately or generate tags for deep analysis
- **Advanced AI Search**: 5 different search methods from exact keywords to premium LLM analysis
- **Flexible Filtering**: Year-based filtering with easy controls
- **Rich Visualizations**: Word clouds, frequency charts, and comprehensive data tables
- **Export Everything**: Search results, visualizations, filtered bibliographies
- **Progress Monitoring**: Real-time logs for all processing operations
- **Responsive Design**: Clean, organized interface that scales to your screen size

## Supported File Formats

### BibTeX Files (.bib)
Standard BibTeX format with fields like:
```bibtex
@article{example2023,
    title={Example Paper Title},
    author={Author, First and Second, Author},
    journal={Example Journal},
    year={2023},
    abstract={This is the abstract of the paper...},
    doi={10.1234/example.2023}
}
```

### CSV Files (.csv)
CSV format with bibliography data columns:
```csv
id,title,author,year,journal,abstract,doi
1,"Example Paper Title","Author, First and Second, Author",2023,"Example Journal","This is the abstract...",10.1234/example.2023
```

**Required CSV columns:**
- `title`: Paper title
- `author`: Author names
- `abstract`: Paper abstract (essential for tag generation)

**Optional CSV columns:**
- `id`: Unique identifier (auto-generated if missing)
- `year`: Publication year
- `journal`: Journal name
- `doi`: Digital Object Identifier
- Any other bibliographic fields

**Year Filtering:**
Both formats support year filtering. Years are extracted intelligently from various formats:
- `2023` (standard)
- `2023-01-01` (date format)
- `c2023` (circa format)
- Only years 1000-2100 are considered valid

## Bibliography Data Quality

### Improving Incomplete Bibliographies with Publication Enricher

For incomplete bibliographies missing abstracts, DOIs, or other metadata, consider using the **[Publication Enricher](https://github.com/ChrisOldmeadow/publication-enricher)** project to clean and enrich your data before analysis:

**Publication Enricher Features:**
- **Missing Field Detection**: Identifies incomplete entries lacking abstracts, DOIs, journals, etc.
- **Automated Enrichment**: Fetches missing metadata from CrossRef, PubMed, and other sources
- **Data Validation**: Verifies existing fields and corrects formatting issues
- **Duplicate Detection**: Identifies and helps merge duplicate entries
- **Format Conversion**: Converts between BibTeX, CSV, RIS, and other formats
- **Quality Reports**: Generates detailed reports on bibliography completeness

**Recommended Workflow:**
1. **Clean your bibliography** with Publication Enricher to add missing abstracts and metadata
2. **Analyze the enriched data** with Bibtex Analyzer GPT for AI-powered tagging and visualization
3. **Export results** in your preferred format with complete bibliographic information

**Example Integration:**
```bash
# Step 1: Enrich incomplete bibliography
publication-enricher enrich incomplete_refs.bib --output enriched_refs.csv --add-abstracts --verify-dois

# Step 2: Analyze enriched bibliography with year filtering
bibtex-analyzer analyze enriched_refs.csv --output tagged_results.csv --min-year 2020

# Step 3: Generate visualizations
bibtex-analyzer visualize tagged_results.csv --wordcloud both
```

This workflow ensures maximum data quality and more accurate tag generation, since abstracts are essential for meaningful AI analysis.

## Advanced Configuration

### Environment Variables

You can configure the application using the following environment variables in your `.env` file:

```env
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
DEFAULT_MODEL=gpt-3.5-turbo  # Default model for tag generation
METHODS_MODEL=gpt-4          # Model for statistical methods
LOG_LEVEL=INFO               # Logging level (DEBUG, INFO, WARNING, ERROR)
UPLOAD_FOLDER=./uploads      # Directory for file uploads
PORT=8050                    # Port for the dashboard
DEBUG=False                  # Run in debug mode
```

### Customizing Tag Generation

You can customize the tag generation prompt by creating a `prompts` directory in your project root and adding a `tag_generation.md` file with your custom prompt. The default prompt includes instructions for generating relevant, specific tags for academic papers.

## Linking Staff Results to the HMRI Affiliate Index

If you maintain the `HMRI affiliate index.csv` file in the project root, you can merge dashboard staff downloads with the official roster using the helper script:

```bash
# Export the staff summary from the dashboard, then run:
python link_staff_affiliates.py staff_summary.csv \
    --index "HMRI affiliate index.csv" \
    --output staff_summary_with_names.csv
```

The script performs a right join on `staff_id` (NumberPlate), adds first/last name plus faculty/school columns, and writes a single row per staff member. Omit `--output` to generate `<original>_with_names.csv` alongside your download.

> Tip: the exported staff summary already contains the `llm_summary` and `llm_focus_topics` columns once you generate the LLM blurbs in the dashboard, so the merged file keeps both the narrative and the official name information.

## Examples

### Example 1: Basic Usage

```bash
# Analyze a BibTeX file and generate tags
bibtex-analyzer analyze papers.bib --output results.csv

# Analyze a CSV file and generate tags
bibtex-analyzer analyze bibliography.csv --output results.csv

# Launch the dashboard to explore results
bibtex-analyzer dashboard
```

### Example 2: Advanced Analysis with Year Filtering

```bash
# Analyze recent papers (2020-2023) with better tag quality
bibtex-analyzer analyze papers.bib \
    --output recent_papers.csv \
    --min-year 2020 \
    --max-year 2023 \
    --tag-samples 100 \
    --model gpt-4

# Generate both PNG and HTML word clouds
bibtex-analyzer visualize recent_papers.csv --wordcloud both
```

### Example 3: Semantic Search for Topic Discovery

```bash
# Search for papers related to chronic fatigue syndrome
bibtex-analyzer search "chronic fatigue syndrome" papers.csv \
    --output cfs_papers.csv \
    --methods exact fuzzy semantic \
    --semantic-threshold 0.7 \
    --max-results 20

# Find papers about machine learning in healthcare using only semantic search
bibtex-analyzer search "machine learning healthcare diagnosis" papers.csv \
    --methods semantic \
    --semantic-threshold 0.75

# Search with fuzzy matching for typos
bibtex-analyzer search "artifical inteligence" papers.csv \
    --methods fuzzy \
    --fuzzy-threshold 70
```

### Example 4: CSV Analysis with Filtering

```bash
# Process a large CSV bibliography, filtering recent papers only
bibtex-analyzer analyze large_bibliography.csv \
    --output filtered_results.csv \
    --min-year 2022 \
    --tag-samples 50 \
    --subset-size 200
```

### Example 5: Docker Deployment

You can also run the dashboard in a Docker container:

```bash
# Build the Docker image
docker build -t bibtex-analyzer .

# Run the container
docker run -p 8050:8050 -e OPENAI_API_KEY=your_api_key_here bibtex-analyzer
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   - Ensure you've set the `OPENAI_API_KEY` environment variable
   - Check that your API key has sufficient credits

2. **Processing Large Files**
   - For large bibliographies, use the `--subset-size` parameter
   - Consider running on a machine with more memory

3. **Word Cloud Generation**
   - If the word cloud is empty, check that tags were generated correctly
   - Try adjusting the minimum word frequency

## Contributing

Cntributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Open a pull request

## License

MIT

## Acknowledgments

- OpenAI for the GPT models
- WordCloud and Plotly for visualization
