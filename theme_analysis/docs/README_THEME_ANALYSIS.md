# Theme Analysis - Complete Usage Guide

**Separate module for strategic research theme analysis**

This module provides command-line tools for batch analysis of institutional research themes. It is **independent from the dashboard** and uses:
- Semantic + LLM hybrid search for theme matching
- SCImago-style institutional scoring (0-100 scale)
- Field-normalized citation metrics
- Staff aggregations per theme
- Data quality tracking with missing data exclusions

---

## Quick Start

### 1. Set up your environment

```bash
# Ensure OpenAI API key is set
export OPENAI_API_KEY="your-key-here"

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Enrich with OpenAlex (optional but recommended)

```bash
python scripts/enrich_with_openalex.py \
  --input data/institutional_publications.csv \
  --output data/institutional_publications_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com
```

**What it does:**
- Looks up publications by DOI in OpenAlex
- Fetches field-weighted citation metrics (percentile-based FWCI approximation)
- Gets open access status and collaboration data
- Caches responses (~20-30 min first time, <1 min subsequent runs)

**Why recommended:**
- Global benchmarks (more accurate)
- Fills missing data gaps
- Reduces paper exclusions due to missing metrics

**If skipped:** More papers will be excluded from scoring due to missing required data.

**Note:** `calculate_baselines.py` exists as an alternative approach for field normalization but is NOT part of the default workflow.

### 3. Define your themes

Edit `themes.yaml` to define your strategic research priorities.

**Option A: Flat structure (simple themes):**

```yaml
themes:
  - id: dementia_intervention
    name: "Dementia Awareness & Intervention"
    narrative: |
      Research focused on Alzheimer's disease, dementia, brain degeneration...
    semantic_threshold: 0.5
    max_candidates: 100          # Hybrid candidate cap
    max_results: 50              # Hybrid output cap
    min_llm_relevance: 6.0
    # Optional overrides:
    # semantic_only: true                # Skip GPT rerank for this theme
    # prompt_on_overflow: true           # Ask before truncating candidate pool
    # semantic_max_candidates: 300       # Separate cap when semantic_only is true
    # semantic_max_results: 200          # Semantic-only result limit (default unlimited)
```

**Option B: Hierarchical structure (topics within themes):**

If you have themes with multiple topics that need individual scoring and ranking:

```yaml
themes:
  # Parent Theme 1: Environmental Health - Topic 1
  - id: env_health_1_housing_food_water
    name: "Housing, Food, and Water Access"
    parent_theme: "Environmental Health and Disasters"
    theme_number: 1
    topic_number: 1
    narrative: |
      Research on housing quality, food security, and water access...
    semantic_threshold: 0.5
    max_candidates: 100
    max_results: 50
    min_llm_relevance: 6.0

  # Parent Theme 1: Environmental Health - Topic 2
  - id: env_health_2_climate_health
    name: "Climate Change and Health Impacts"
    parent_theme: "Environmental Health and Disasters"
    theme_number: 1
    topic_number: 2
    narrative: |
      Research on health impacts of climate change...
    # ... continue for all topics
```

See `themes_example_hierarchical.yaml` for a complete hierarchical example and `docs/hierarchical_themes_guide.md` for detailed guidance.

### 4. Run theme search

```bash
python scripts/run_theme_search.py \
  --dataset data/institutional_publications_enriched.csv \
  --themes themes.yaml \
  --output results/theme_analysis_2025/
  # Optional flags:
  # --max-candidates 150   # Override per-theme candidate cap
  # --candidate-prompt     # Ask before truncating large candidate pools
  # --semantic-only        # Disable GPT rerank (fast, no API usage)
```

**Or without enrichment:**
```bash
python scripts/run_theme_search.py \
  --dataset data/institutional_publications.csv \
  --themes themes.yaml \
  --output results/theme_analysis_2025/
```

**What it does:**
1. Loads and deduplicates publications
2. Checks data completeness for each paper
3. Embeds papers (with caching)
4. For each theme:
   - Runs semantic search across all papers
   - Analyzes top candidates with GPT-4o-mini
   - Filters by LLM relevance threshold
   - Calculates SCImago scores (only papers with complete data)
   - Papers with missing data excluded but flagged in outputs
   - Aggregates by staff member
5. Generates cross-theme comparison with coverage statistics

**Data Quality:** Papers missing ANY required metric are excluded from scoring (all-or-nothing approach). After reviewing exclusion rates, imputation strategies can be considered.

**Controlling GPT usage**

- `--max-candidates` globally raises/lowers how many papers per theme move to GPT (falls back to each theme’s `max_candidates`, default 100).
- `--candidate-prompt` pauses when the semantic filter finds more matches than the cap so you can keep the cap, send all matches, or enter a custom limit.
- `--semantic-only` skips GPT entirely and relies on embedding similarity (you can also set `semantic_only: true` per theme). In this mode, the generic `max_candidates`/`max_results` caps are ignored unless you provide the semantic-specific variants (`semantic_max_candidates`, `semantic_max_results`) or a CLI override. The pipeline still emits placeholder relevance scores derived from embeddings for downstream compatibility.
- `--ignore-max-limits` ignores any `max_candidates`/`max_results` defined in `themes.yaml`, letting both hybrid and semantic searches reuse every match that clears the thresholds (unless `--max-candidates` is supplied).
- Per-theme overrides: add `max_candidates`, `semantic_only`, or `prompt_on_overflow` keys inside a theme entry to customize behavior for specific narratives.

**Processing time:** ~5-10 minutes for 3 themes × 1000 publications (with caching)

### 5. Analyze hierarchical results (if using hierarchical structure)

If you used Option B (topics within themes), run the hierarchical analysis to rank topics within each parent theme:

```bash
python scripts/analyze_hierarchical_themes.py \
  --results results/theme_analysis_2025/theme_comparison.csv \
  --output results/theme_analysis_2025/
```

**What it does:**
- Groups topics by parent theme
- Ranks topics 1-5 within each parent theme
- Generates parent theme-level summaries
- Creates detailed hierarchical report

**Additional outputs:**
- `topic_rankings_by_theme.csv` - Topics with rank within parent theme
- `parent_theme_summary.csv` - Comparison of parent themes
- `hierarchical_theme_report.md` - Detailed markdown report

**See:** `docs/hierarchical_themes_guide.md` for complete documentation on hierarchical analysis.

---

## Output Files

```
results/theme_analysis_2025/
├── theme_comparison.csv          # Rankings and scores for all themes
│
├── dementia_intervention/
│   ├── papers.csv                # All papers for this theme
│   └── staff_summary.csv         # Staff aggregations
│
└── [other theme folders]/
```

### theme_comparison.csv

| Column | Description |
|--------|-------------|
| `publications` | Total papers matching theme |
| `publications_scored` | Papers with complete data (used in scoring) |
| `publications_excluded` | Papers excluded due to missing data |
| `data_completeness_rate` | % papers with complete data |
| `theme_score` | Overall score 0-100 (from scored papers only) |
| `research_score` | Research performance (0-100) |
| `societal_score` | Altmetric coverage (percentage of papers with attention) |
| `unique_staff` | Number of contributing researchers |
| `q1_percentage` | % papers in top-quartile journals |
| `normalized_impact` | Citations relative to field average |
| `excellence_rate` | % papers in top 10% most cited |
| `altmetric_coverage` | % papers with social media attention |
| `avg_altmetric_score` | Mean Altmetric score for papers with attention |

### [theme]/papers.csv

Individual papers with:
- All original publication fields
- OpenAlex enrichment fields (if enriched)
- `llm_relevance_score` (0-10) - GPT assessment of fit
- `llm_reasoning` - Explanation of relevance
- `llm_key_concepts` - Matched concepts
- `hybrid_score` - Combined embedding + LLM score
- `excluded_from_scoring` - True if missing required data
- `missing_metrics` - List of missing required fields
- `data_quality_score` - % of required metrics present (0-1)

### [theme]/staff_summary.csv

Staff aggregations with:
- `paper_count` - Total publications in theme
- `avg_relevance` - Average LLM relevance score
- `high_quality_papers` - Count with score ≥ 8.0
- `avg_hybrid_score` - Average combined score

---

## Required Data for Scoring

Papers need **ALL** of these to contribute to theme scores:
1. **Field-normalized impact** - OpenAlex FWCI approximation
2. **Journal quality** - Clarivate_Quartile_Rank OR SJR_Best_Quartile
3. **Citations** - Citations_Scopus OR Citations_WoS OR OpenAlex
4. **Open access** - Ref_PMC_ID OR openalex_is_oa
5. **International collaboration** - Countries OR openalex_is_international
6. **Altmetrics** - Altmetrics_Score
7. **Author position** - First_Author, Last_Author

Missing ANY of these → paper excluded (but still listed in outputs with flags).

**Strategy:** Run first with all-or-nothing approach, review exclusion rates, then decide on imputation if needed.

---

## Adjusting Parameters

If you get too many or too few results, edit `themes.yaml`:

```yaml
# Too many irrelevant papers? Increase:
semantic_threshold: 0.6       # Stricter embedding match
min_llm_relevance: 7.0        # Only highly relevant papers

# Too few results? Decrease:
semantic_threshold: 0.45      # Broader semantic net
min_llm_relevance: 5.5        # Include tangentially related

# Want more GPT analysis? Increase:
max_candidates: 150           # More papers analyzed (costs more)
```

Then re-run step 4. **Caching ensures you don't recompute unchanged embeddings/analyses!**

---

## Annual Updates

When you get new publication data:

```bash
# 1. Enrich new data with OpenAlex
python scripts/enrich_with_openalex.py \
  --input data/institutional_publications_2026.csv \
  --output data/institutional_publications_2026_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com

# 2. Run theme search with new data
python scripts/run_theme_search.py \
  --dataset data/institutional_publications_2026_enriched.csv \
  --themes themes.yaml \
  --output results/theme_analysis_2026/

# 3. Compare year-over-year (optional utility to be created)
# Shows changes in theme scores, volumes, and data completeness
```

This shows how theme scores, publication volumes, and data completeness changed year-over-year.

---

## Cost Management

### First Run
- **OpenAlex enrichment:** Free (rate-limited API)
- **Embeddings:** ~$0.10 per 1000 publications
- **LLM analysis:** ~$0.50 per 100 papers (GPT-4o-mini)
- **Total for 1000 pubs × 3 themes:** ~$15-20

### Subsequent Runs
- **Same data:** $0 (fully cached)
- **New themes on same data:** May reuse cached analyses if papers overlap
- **Annual updates (10% new):** ~$2-3

**Caching locations:**
- `.embeddings_cache/` - Text embeddings
- `.llm_relevance_cache/` - GPT relevance analyses
- `data/openalex_cache.json` - OpenAlex API responses

---

## Troubleshooting

### "High exclusion rate (>30%)"
- Most likely: missing field-normalized impact data
- Solution: Run OpenAlex enrichment to fill gaps
- Review `missing_metrics` column in papers.csv
- Consider imputation strategy after understanding patterns

### "No results found for theme X"
- Narrative may be too specific or threshold too high
- Try lowering `semantic_threshold` to 0.45
- Check if papers have abstracts populated

### "Too many irrelevant papers"
- Increase `min_llm_relevance` to 7.0
- Increase `semantic_threshold` to 0.55
- Refine narrative to be more specific

### "Processing is slow"
- First run always slower (computing embeddings + LLM)
- Check cache directories exist
- Reduce `max_candidates` to analyze fewer papers

### "Theme score seems low"
- Check data completeness rate in theme_comparison.csv
- Review which papers were excluded (check `missing_metrics` column)
- Consider OpenAlex enrichment if not already done
- Verify citation data populated in publications

---

## Advanced Usage

### Filter by date range

```bash
python scripts/run_theme_search.py \
  --dataset data/institutional_publications_enriched.csv \
  --themes themes.yaml \
  --output results/themes_2020_2024/ \
  --min-year 2020 \
  --max-year 2024
```

### Custom scoring weights

Edit `theme_analysis/scoring.py` to adjust component weights. Default research mix:

```python
# Defaults (output capped at 250 pubs; normalized impact capped at 2× world avg)
research_score = (
    0.25 * output_normalized +
    0.15 * intl_collab_rate +
    0.20 * q1_rate +
    0.10 * min(normalized_impact / 2, 1.0) +
    0.15 * excellence_rate +
    0.10 * leadership_rate +
    0.05 * open_access_rate
)
```

To emphasize citations, bump the normalized impact weight and reduce another component, e.g.:

```python
research_score = (
    0.25 * output_normalized +
    0.15 * intl_collab_rate +
    0.15 * q1_rate +                 # reduced
    0.15 * min(normalized_impact / 2, 1.0) +  # increased
    0.15 * excellence_rate +
    0.10 * leadership_rate +
    0.05 * open_access_rate
)
```

Societal score currently equals the altmetric coverage percentage (percentage of papers with a non-zero Altmetric score); the raw `avg_altmetric_score` column is included for context.

---

## Documentation

- **`scripts/QUICK_START.md`** - Quick reference for HMRI data
- **`docs/research_topic_search_and_score_strategy.md`** - Overall strategy and architecture
- **`docs/theme_scoring_methodology.md`** - Complete scoring algorithm details
- **`docs/hierarchical_themes_guide.md`** - Guide for topics within themes (hierarchical structure)
- **`docs/data_quality_guide.md`** - Data quality best practices and troubleshooting
- **`themes.yaml`** - Parameter guidelines and examples
- **`themes_example_hierarchical.yaml`** - Example hierarchical structure with 3 themes × 5 topics

---

## File Structure

```
bibtex-analyser-llm/
├── theme_analysis/              # Separate module (independent from dashboard)
│   ├── __init__.py
│   ├── baselines.py            # Baseline calculation (optional)
│   ├── scoring.py              # SCImago scoring logic
│   └── pipeline.py             # Theme search orchestration
│
├── scripts/
│   ├── enrich_with_openalex.py        # Step 2: Enrich with OpenAlex
│   ├── calculate_baselines.py         # Optional: Alternative normalization
│   ├── run_theme_search.py            # Step 4: Run theme search
│   ├── analyze_hierarchical_themes.py # Step 5: Hierarchical analysis (optional)
│   ├── QUICK_START.md                 # Quick reference
│   └── README_THEME_ANALYSIS.md       # This file
│
├── themes.yaml                        # Theme definitions
├── themes_example_hierarchical.yaml   # Example: 3 themes × 5 topics
│
├── bibtex_analyzer/            # Existing package (unchanged)
│   ├── dashboard.py            # Dashboard NOT affected
│   └── semantic_search.py      # Reused by theme analysis
│
└── results/                    # Theme analysis outputs
    └── theme_analysis_2025/
```

---

## Questions?

See the main documentation in `docs/` or check existing issues on GitHub.
