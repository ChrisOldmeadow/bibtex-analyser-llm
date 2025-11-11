# Theme Analysis Module - Implementation Summary

## What Was Built

A **complete, standalone module** for strategic research theme analysis, independent from the existing dashboard.

### Location
- **Module:** `theme_analysis/` (new)
- **Scripts:** `scripts/` (calculate_baselines.py, run_theme_search.py)
- **Config:** `themes.yaml`
- **Docs:** `scripts/QUICK_START.md`, `scripts/README_THEME_ANALYSIS.md`, `docs/`

---

## Key Features

✅ **SCImago-style institutional scoring** (0-100 scale)
- Research performance (65%): Output (normalized to ~90th percentile of publication counts), Q1%, citations, excellence, leadership, OA
- Societal impact (35%): Altmetric coverage (with average score reported for context)

✅ **Field normalization**
- Uses OpenAlex FWCI (Field-Weighted Citation Impact) via DOI lookup
- Global benchmarks for fair cross-discipline comparison
- Alternative: Dataset baselines (optional tool, not default)

✅ **Automatic deduplication**
- Handles HMRI CSV with duplicate rows per staff member
- Preserves all `NumberPlate` values in `all_staff_ids` arrays

✅ **Cost optimization**
- Uses existing `.embeddings_cache/` (no re-embedding!)
- Uses existing `.llm_relevance_cache/` where possible
- OpenAlex API is free with polite rate limiting
- Only pays for new GPT analyses

✅ **Data quality tracking**
- All-or-nothing exclusion: papers need ALL required metrics to score
- Excluded papers still listed with flags showing what's missing
- Completeness rates reported per theme
- Assess-then-impute strategy: review patterns before deciding on imputation

✅ **Staff aggregations**
- Papers per theme per researcher
- Average relevance scores
- High-quality paper counts

---

## Architecture

```
theme_analysis/                    # NEW MODULE (separate from dashboard)
├── __init__.py
├── baselines.py                   # Dataset baseline calculations
├── scoring.py                     # SCImago scoring logic
└── pipeline.py                    # Theme search orchestration

scripts/                           # CLI TOOLS
├── enrich_with_openalex.py        # Step 1: Enrich with OpenAlex (optional)
├── calculate_baselines.py         # Optional: Alternative normalization (not default)
├── run_theme_search.py            # Step 2: Run theme analysis
├── analyze_hierarchical_themes.py # Step 3: Hierarchical analysis (optional)
├── QUICK_START.md                 # Quick reference for HMRI data
└── README_THEME_ANALYSIS.md       # Complete documentation

themes.yaml                        # Theme definitions (flat or hierarchical)
themes_example_hierarchical.yaml   # Example: 3 themes × 5 topics

docs/
├── research_topic_search_and_score_strategy.md   # Overall strategy
├── theme_scoring_methodology.md                  # Scoring algorithm details
├── hierarchical_themes_guide.md                  # Guide for topics within themes
└── data_quality_guide.md                         # Data quality best practices
```

**Dashboard untouched** - `bibtex_analyzer/dashboard.py` remains unchanged.

---

## How It Works

### 1. OpenAlex Enrichment (optional but recommended, ~20-30 min first run)

```bash
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com
```

**Processing:**
1. Loads CSV using `BibtexProcessor` → automatic deduplication
2. Looks up each publication by DOI in OpenAlex
3. Fetches field-weighted citation impact (FWCI proxy via percentile)
4. Gets open access status and collaboration flags
5. Caches all responses for future runs
6. Saves enriched CSV with new columns

**Output:** Enriched CSV with OpenAlex metrics + coverage report

**Why recommended:**
- Global benchmarks (more accurate than dataset-only baselines)
- Fills missing data gaps (OA status, collaboration info)
- Reduces paper exclusions due to missing metrics
- Free API with polite rate limiting

---

### 2. Theme Search (batch across all themes)

```bash
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/hmri_themes_2025/
```

**Processing per theme:**
1. **Deduplicate:** Load CSV → merge duplicate rows → collect staff IDs
2. **Data Quality Check:** Check each paper for required metrics → flag incomplete papers
3. **Embed:** Prepare title+abstract text → check cache → embed if new
4. **Search:** Semantic similarity between query narrative and all papers
5. **Filter:** Keep top N by embedding score (e.g., top 100)
6. **Analyze:** GPT-4o-mini scores relevance (0-10) with reasoning
7. **Score:** Calculate SCImago-style theme score (only papers with complete data)
8. **Aggregate:** Group by staff ID for researcher-level summaries
9. **Export:** Papers CSV (with exclusion flags), staff summary CSV, coverage stats per theme

**Cross-theme comparison:**
- Ranks all themes by overall score
- Shows publications (total/scored/excluded), data completeness rates
- Displays all component scores and metrics side-by-side

**Optional Step 3: Hierarchical analysis (if topics within themes):**
```bash
python scripts/analyze_hierarchical_themes.py \
  --results results/hmri_themes_2025/theme_comparison.csv \
  --output results/hmri_themes_2025/
```

- Groups topics by parent theme
- Ranks topics within each theme (1-5)
- Generates parent theme summaries
- Creates detailed hierarchical report

---

## Data Flow: HMRI CSV Handling

**Input:** `HMRI Pub Abstracts_20250703.csv` (173MB)
- Contains duplicate rows (one per staff member per publication)
- Example: 5-author paper → 5 rows with different `NumberPlate`

**Deduplication (automatic via BibtexProcessor):**
```
Row 1: Publication_ID=12345, NumberPlate=ABC123
Row 2: Publication_ID=12345, NumberPlate=DEF456
Row 3: Publication_ID=12345, NumberPlate=GHI789
         ↓
Merged: Publication_ID=12345, all_staff_ids=['ABC123', 'DEF456', 'GHI789']
```

**Embedding cache reuse:**
- Cache keys based on: `model:text_content` (MD5 hash)
- Your existing cache from dashboard already covers deduplicated papers
- No re-embedding needed for existing papers!

**Data quality checking:**
- Each unique publication checked for 7 required metrics
- Papers missing ANY metric are flagged as `excluded_from_scoring: True`
- Missing metrics listed for transparency

**Scoring:**
- Only papers with complete data contribute to theme scores
- Each unique publication scored once
- Staff aggregation uses `all_staff_ids` to credit all contributors
- Excluded papers still appear in outputs with flags

---

## Output Structure

```
results/hmri_themes_2025/
├── theme_comparison.csv              # Cross-theme rankings
│   └── Columns: publications, publications_scored, publications_excluded,
│                data_completeness_rate, theme_score, research_score, societal_score,
│                q1_percentage, normalized_impact, excellence_rate, etc.
│
├── dementia_intervention/
│   ├── papers.csv
│   │   └── Unique papers with all_staff_ids array, LLM scores, reasoning,
│          excluded_from_scoring flag, missing_metrics list, data_quality_score
│   └── staff_summary.csv
│       └── Aggregated by NumberPlate: paper_count, avg_relevance, etc.
│
└── [other theme folders]/
```

---

## Cost Estimates

### First Run (5 themes, ~30K unique publications)
- **OpenAlex enrichment:** $0 (free API, ~20-30 min first run)
- **Embeddings:** $0 (uses existing cache)
- **LLM analysis:** 5 themes × 100 candidates × $0.005/paper ≈ **$2.50**
- **Total:** ~$2.50

### Subsequent Runs (same themes, same data)
- **Everything cached:** $0

### Annual Updates (10% new publications)
- **OpenAlex enrichment:** $0 (cached for existing, new lookups free)
- **New embeddings:** ~$0.30
- **New LLM analyses:** ~$0.50
- **Total:** ~$0.80

---

## Parameter Tuning

**In `themes.yaml` for each theme:**

```yaml
semantic_threshold: 0.5       # Embedding similarity (0-1)
  # Higher = stricter, fewer but more relevant papers
  # Lower = broader, more papers but some irrelevant
  # Recommended: 0.45-0.6

max_candidates: 100           # Papers to send to GPT
  # More = higher cost but better coverage
  # Typical: 80-150

max_results: 50               # Final result set size
  # Can be same as max_candidates or smaller

min_llm_relevance: 6.0        # LLM score threshold (0-10)
  # 8+ = highly relevant core research
  # 6-7 = relevant with connection
  # 5-6 = tangentially related
  # Recommended: 6.0-6.5
```

---

## Integration Points

**Reuses from existing codebase:**
- `bibtex_analyzer.semantic_search.HybridSemanticSearcher` - hybrid search logic
- `bibtex_analyzer.bibtex_processor.BibtexProcessor` - deduplication
- `.embeddings_cache/` - OpenAI embeddings
- `.llm_relevance_cache/` - GPT relevance analyses

**No changes to:**
- Dashboard (`bibtex_analyzer/dashboard.py`)
- Tag generator (`bibtex_analyzer/tag_generator.py`)
- Visualization modules

---

## Testing Checklist

Before running on full dataset, test with a small sample:

1. ✅ **Check deduplication:**
   ```bash
   python -c "
   from bibtex_analyzer.bibtex_processor import BibtexProcessor
   p = BibtexProcessor()
   entries = p.load_entries('HMRI Pub Abstracts_20250703.csv', deduplicate=True)
   print(f'Unique: {len(entries)}')
   print(f'Original: {p.deduplication_stats[\"total_entries\"]}')
   print(f'Sample all_staff_ids:', entries[0].get('all_staff_ids', [])[:5])
   "
   ```

2. ✅ **Check cache reuse:**
   ```bash
   ls .embeddings_cache/*.pkl | wc -l
   ls .llm_relevance_cache/*.json | wc -l
   ```

3. ✅ **Test OpenAlex enrichment (optional):**
   ```bash
   python scripts/enrich_with_openalex.py \
     --input "HMRI Pub Abstracts_20250703.csv" \
     --output data/test_enriched.csv \
     --cache data/openalex_cache.json \
     --email your.email@example.com

   # Check coverage report
   cat data/openalex_coverage_report.json
   ```

4. ✅ **Test theme search (1 theme):**
   ```bash
   # Edit themes.yaml to have only 1 theme
   python scripts/run_theme_search.py \
     --dataset data/test_enriched.csv \
     --themes themes.yaml \
     --output results/test_run/
   ```

5. ✅ **Verify outputs:**
   - Check `theme_comparison.csv` has data quality columns (publications_scored, publications_excluded, data_completeness_rate)
   - Check `papers.csv` has `all_staff_ids` array and exclusion flags (excluded_from_scoring, missing_metrics)
   - Check `staff_summary.csv` aggregations look correct
   - Review completeness rate - if <70%, investigate missing_metrics patterns

---

## Next Steps

1. **Run OpenAlex enrichment** (optional but recommended, ~20-30 minutes first time)
2. **Review and customize `themes.yaml`** with your strategic themes
   - Use flat structure for simple themes OR
   - Use hierarchical structure for topics within themes (see `themes_example_hierarchical.yaml`)
3. **Run theme search** for all themes/topics (~5-25 minutes depending on structure)
4. **If hierarchical: Analyze topic rankings** within each parent theme (<1 minute)
5. **Review data quality reports** - check completeness rates in `theme_comparison.csv`
6. **Examine exclusion patterns** - review `missing_metrics` in `papers.csv` files
7. **Decide on Phase 2 strategy** - if exclusion rate >30%, consider imputation approaches
8. **Adjust parameters** if needed and re-run (fast with caching!)

---

## Documentation

- **Quick start:** `scripts/QUICK_START.md`
- **Detailed usage:** `scripts/README_THEME_ANALYSIS.md`
- **Strategy:** `docs/research_topic_search_and_score_strategy.md`
- **Scoring algorithm:** `docs/theme_scoring_methodology.md`
- **Hierarchical themes:** `docs/hierarchical_themes_guide.md` (for topics within themes)
- **Data quality:** `docs/data_quality_guide.md`
- **Examples:** `themes_example_hierarchical.yaml`

---

## Questions & Support

**Common issues:**
- Module not found → ensure in project directory, install dependencies
- No results → lower thresholds or broaden narrative
- High exclusion rate (>30%) → run OpenAlex enrichment to fill missing data gaps
- Deduplication issues → check `Publication_ID` field exists
- Cache not working → check `.embeddings_cache/` directory exists
- Low theme scores → check data completeness rate and review excluded papers

**Data quality troubleshooting:**
- Review `missing_metrics` column in `papers.csv` to identify patterns
- Check `data/openalex_coverage_report.json` for enrichment success rates
- If specific metrics consistently missing, consider alternative data sources
- Calculate_baselines.py available as optional tool for alternative normalization

See troubleshooting sections in QUICK_START.md and README_THEME_ANALYSIS.md.
