# Bibtex Analyzer GPT

A powerful tool for analyzing and visualizing BibTeX bibliographies using AI-powered tagging and interactive visualizations. This tool helps researchers, academics, and anyone working with academic papers to better understand and explore their reference collections.

## Features

- **BibTeX Processing**: Extract and process BibTeX entries from `.bib` files
- **AI-Powered Tagging**: Generate topic tags using OpenAI's GPT models
- **Interactive Visualizations**:
  - Optional word clouds (static PNG and/or interactive HTML)
  - Clickable word clouds that link to relevant papers
  - Tag frequency analysis
  - Interactive tag networks
- **Filtering & Searching**: Easily filter and search through your bibliography
- **Export Options**: Save results in multiple formats (CSV, HTML, PNG, PDF)
- **Interactive Dashboard**: Web-based interface for exploring your bibliography with:
  - File upload and processing
  - Real-time word cloud generation
  - Tag frequency analysis
  - Data table with filtering and sorting
  - Export functionality
  - Customizable visualization options

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
# Analyze a BibTeX file and generate tags with optional word cloud
bibtex-analyzer analyze papers.bib --output results.csv --tag-samples 50 --wordcloud

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

# Analyze a BibTeX file and generate tags
# Using default settings (30 samples for tag generation, tag all entries)
bibtex-analyzer analyze input.bib --output tagged_papers.csv

# Customize tag generation and processing
bibtex-analyzer analyze input.bib \
    --output tagged_papers.csv \
    --tag-samples 50 \
    --subset-size 200

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
from bibtex_analyzer import process_bibtex_file, TagGenerator

# Process a BibTeX file and get tagged entries
entries = process_bibtex_file("input.bib", output_file="output/tagged.csv")

# Generate tags for a list of entries
tag_generator = TagGenerator(api_key="your_api_key")
tags = tag_generator.generate_tags_for_abstracts(entries)

# Assign tags to entries
tagged_entries = tag_generator.assign_tags_to_abstracts(entries, tags)
```

### Command Line Options

```
usage: bibtex-analyzer [-h] {analyze,visualize,dashboard} ...

Bibtex Analyzer - Analyze and visualize BibTeX bibliographies

positional arguments:
  {analyze,visualize,dashboard}
    analyze             Analyze a BibTeX file and generate tags
    visualize           Generate visualizations from tagged data
    dashboard           Launch interactive web dashboard

options:
  -h, --help            show this help message and exit
```

#### Analyze Command

```
usage: bibtex-analyzer analyze [-h] [--output OUTPUT] [--tag-samples TAG_SAMPLES]
                              [--subset-size SUBSET_SIZE] [--model MODEL]
                              [--methods-model METHODS_MODEL]
                              input

positional arguments:
  input                 Input BibTeX file

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output CSV file (default: tagged_abstracts.csv)
  --tag-samples TAG_SAMPLES
                        Number of samples to use for tag generation (default: 30)
  --subset-size SUBSET_SIZE
                        Number of random entries to tag (0 for all) (default: 100)
  --model MODEL         Model to use for tag generation (default: gpt-3.5-turbo)
  --methods-model METHODS_MODEL
                        Model to use for statistical methods (default: gpt-4)
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

The interactive dashboard provides a user-friendly web interface for exploring your bibliography. Here's what you can do:

### 1. Upload & Process
- Drag and drop your BibTeX file (.bib) or click to browse
- Upload your BibTeX file
- Configure tag generation settings:
  - Tag sample size
  - Maximum entries to tag
  - Model selection (GPT-3.5 Turbo or GPT-4)
- Monitor processing progress with real-time logs

### 2. Word Cloud Customization
- Adjust word cloud settings:
  - Maximum number of words
  - Color schemes (Viridis, Plasma, Inferno, Magma, Cividis, Rainbow)
  - Background colors (White, Black, Light Gray, Dark)
- Download word cloud as PNG or interactive HTML

### 3. Data Table
- View all entries with key information:
  - Year
  - Title
  - Authors
  - Journal
  - DOI
  - Abstract
  - Volume
  - Issue
  - Pages
  - Keywords
- Filter entries by tags using a searchable dropdown
- Download BibTeX file with two options:
  - Download All Entries: Get the complete bibliography
  - Download Filtered Entries: Only get entries matching your selected tags
- Entries include all available metadata fields and generated keywords


### Visualization Options
- **Word Cloud**: Generate beautiful word clouds from your paper tags
  - Toggle between static (PNG) and interactive (HTML) versions
  - Customize colors, size, and number of words
  - Click on words to see related papers
- **Tag Frequencies**: Bar chart showing most common tags
  - Hover to see exact counts
  - Click to filter papers by tag

### Data Table
- Sort and filter your bibliography
- Search by title, author, or tags
- View detailed paper information including abstract and tags
- Export filtered results to CSV

### Configuration
- Adjust tag generation settings
- Choose different OpenAI models
- Customize visualization appearance

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

## Examples

### Example 1: Basic Usage

```bash
# Analyze a BibTeX file and generate tags
bibtex-analyzer analyze papers.bib --output results.csv

# Launch the dashboard to explore results
bibtex-analyzer dashboard
```

### Example 2: Advanced Analysis

```bash
# Use more samples for better tag quality
bibtex-analyzer analyze papers.bib \
    --output results.csv \
    --tag-samples 100 \
    --model gpt-4

# Generate both PNG and HTML word clouds
bibtex-analyzer visualize results.csv --wordcloud both
```

### Example 3: Docker Deployment

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
