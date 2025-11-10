# Theme Analysis Pipeline - Quick Run Guide

**Complete workflow for running hierarchical theme analysis on HMRI data**

---

## Prerequisites

```bash
# Ensure you're in the project directory
cd /Users/chris/Projects/bibtex-analyser-llm

# Ensure OpenAI API key is set
export OPENAI_API_KEY="your-key-here"

# Verify themes.yaml is configured with your 16 topics (3 themes)
```

---

## Step 1: Enrich with OpenAlex (Recommended, ~20-30 minutes first time)

This fills missing data gaps and improves data quality:

```bash
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com
```

**Replace `your.email@example.com` with your actual email address.**

**What it does:**
- Automatically deduplicates publications (one row per unique paper)
- Preserves all staff IDs in `all_staff_ids` array
- Looks up each publication in OpenAlex by DOI
- Fetches field-weighted citation impact (FWCI)
- Gets open access status and collaboration flags
- Caches all responses for future runs
- ~30,000-40,000 unique publications from ~100,000 input rows

**Outputs:**
- `data/hmri_enriched.csv` - Enriched, deduplicated publication data
- `data/openalex_cache.json` - API response cache (reused in future runs)
- `data/openalex_coverage_report.json` - Coverage statistics

**Check coverage:**
```bash
cat data/openalex_coverage_report.json
```

---

## Step 2: Run Theme Search (~15-25 minutes for 16 topics)

Analyzes all 16 topics independently:

```bash
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/hmri_themes_2025/
```

**Or if you skipped Step 1 (not recommended):**
```bash
python scripts/run_theme_search.py \
  --dataset "HMRI Pub Abstracts_20250703.csv" \
  --themes themes.yaml \
  --output results/hmri_themes_2025/
```

**What it does:**
- Loads and deduplicates publications (if not already enriched)
- Checks data completeness for each paper (7 required metrics)
- Uses existing embeddings cache (no re-embedding cost!)
- For each of 16 topics:
  - Semantic search across all papers
  - GPT-4o-mini analyzes top candidates
  - Filters by relevance threshold
  - Calculates SCImago-style scores (only papers with complete data)
  - Aggregates by staff member
- Papers with missing data are flagged but still listed in outputs

**Outputs:**
- `results/hmri_themes_2025/theme_comparison.csv` - All 16 topics with scores (flat list)
- `results/hmri_themes_2025/env_health_housing_food_water/papers.csv` - Papers for topic 1
- `results/hmri_themes_2025/env_health_housing_food_water/staff_summary.csv` - Staff aggregations
- ... (15 more topic folders)

**Cost:** ~$1-2 per topic for new GPT analyses (uses cache where possible)

---

## Step 3: Analyze Hierarchical Results (<1 minute)

Groups topics by parent theme and ranks them:

```bash
python scripts/analyze_hierarchical_themes.py \
  --results results/hmri_themes_2025/theme_comparison.csv \
  --output results/hmri_themes_2025/
```

**What it does:**
- Groups 16 topics by parent theme (Environmental Health, Resilient Health, Innovation)
- Ranks topics within each parent theme (1-5 or 1-6)
- Calculates parent theme-level summaries
- Generates detailed hierarchical report

**Outputs:**
- `results/hmri_themes_2025/topic_rankings_by_theme.csv` - Topics ranked within each theme
- `results/hmri_themes_2025/parent_theme_summary.csv` - Comparison of 3 parent themes
- `results/hmri_themes_2025/hierarchical_theme_report.md` - Detailed markdown report

---

## Review Results

### Check Parent Theme Rankings

```bash
cat results/hmri_themes_2025/parent_theme_summary.csv
```

Shows which of your 3 themes has strongest research performance.

### Check Topic Rankings Within Each Theme

```bash
cat results/hmri_themes_2025/topic_rankings_by_theme.csv
```

Shows which topics rank 1-5 (or 1-6) within each parent theme.

### Read Detailed Report

```bash
open results/hmri_themes_2025/hierarchical_theme_report.md
```

Or open in TextMate/VS Code for formatted view.

### Check Data Quality

```bash
# Look for low completeness rates
grep "data_completeness_rate" results/hmri_themes_2025/theme_comparison.csv
```

If any topic has <70% completeness, review the `missing_metrics` column in that topic's papers.csv file.

---

## Troubleshooting

### "No OpenAI API key found"
```bash
export OPENAI_API_KEY="your-key-here"
```

### "Module not found"
```bash
# Install dependencies
pip install -r requirements.txt
```

### "High exclusion rate (>30%)"
- Make sure you ran Step 1 (OpenAlex enrichment)
- Check `data/openalex_coverage_report.json` for coverage statistics
- Review `missing_metrics` column in papers.csv files

### "Processing is slow"
- First run is always slower (computing embeddings + LLM analyses)
- Check that cache directories exist (.embeddings_cache/, .llm_relevance_cache/)
- Subsequent runs will be much faster due to caching

### "Too many/few papers for a topic"
Edit `themes.yaml` and adjust parameters:
- Increase `semantic_threshold` (0.5 → 0.6) for fewer papers
- Decrease `semantic_threshold` (0.5 → 0.45) for more papers
- Increase `min_llm_relevance` (6.0 → 7.0) for only highly relevant papers
- Then re-run Step 2 (caching ensures fast re-runs)

---

## Complete Command Sequence

Copy-paste this entire block to run the full pipeline:

```bash
# Navigate to project
cd /Users/chris/Projects/bibtex-analyser-llm

# Step 1: OpenAlex enrichment (~20-30 min first time)
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com

# Step 2: Theme search (~15-25 min for 16 topics)
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/hmri_themes_2025/

# Step 3: Hierarchical analysis (<1 min)
python scripts/analyze_hierarchical_themes.py \
  --results results/hmri_themes_2025/theme_comparison.csv \
  --output results/hmri_themes_2025/

# View results
echo "=== Parent Theme Summary ==="
cat results/hmri_themes_2025/parent_theme_summary.csv
echo ""
echo "=== Topic Rankings ==="
cat results/hmri_themes_2025/topic_rankings_by_theme.csv
```

**Don't forget to replace `your.email@example.com` with your actual email in Step 1!**

---

## Next Steps After Running

1. **Review hierarchical_theme_report.md** - See detailed rankings and insights
2. **Check data completeness rates** - Identify topics with missing data issues
3. **Examine top-ranked topics** - Review papers.csv for highest-scoring topics
4. **Identify key researchers** - Check staff_summary.csv files per topic
5. **Adjust parameters if needed** - Modify themes.yaml and re-run Step 2

---

## Documentation

- **Complete guide:** `scripts/README_THEME_ANALYSIS.md`
- **Quick reference:** `scripts/QUICK_START.md`
- **Hierarchical themes:** `docs/hierarchical_themes_guide.md`
- **Data quality:** `docs/data_quality_guide.md`
- **Scoring methodology:** `docs/theme_scoring_methodology.md`
