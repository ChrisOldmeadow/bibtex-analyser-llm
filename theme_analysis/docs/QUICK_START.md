# Theme Analysis - Quick Start for HMRI Data

## Your Setup

✅ **Data:** `HMRI Pub Abstracts_20250703.csv` (173MB, contains duplicated rows per staff member)
✅ **Embeddings:** `.embeddings_cache/` (already computed for deduplicated papers)
✅ **API Key:** Set with `export OPENAI_API_KEY="your-key"`

## Important: Data Structure

Your CSV has **duplicate rows** - one per staff member per publication. For example:
- Paper with 5 authors → appears 5 times in CSV
- Each row has a different `NumberPlate` (staff ID)

**The theme analysis pipeline will:**
1. Automatically deduplicate by `Publication_ID`
2. Collect all `NumberPlate` values into `all_staff_ids` array
3. Use existing embeddings cache (already computed on deduplicated papers)
4. Score each unique publication once
5. Aggregate back to staff level for summaries

---

## Step-by-Step Workflow

### 1. Enrich with OpenAlex (optional but recommended, ~20-30 minutes first run)

```bash
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com
```

**What it does:**
- Looks up each publication by DOI in OpenAlex
- Fetches field-weighted citation impact (FWCI proxy via percentile)
- Gets open access status, international collaboration flags
- Adds citation counts for validation
- Caches all responses for future runs (subsequent runs much faster!)

**Why recommended:**
- ✅ Global benchmarks for field-weighted citation impact
- ✅ Fills in missing data (OA status, collaboration info)
- ✅ Already field-normalized (accurate benchmarks)
- ✅ Reduces papers excluded due to missing data

**If you skip this step:** Papers may be excluded from scoring due to missing required metrics (field-normalized impact, OA status, etc.).

**Output:** `data/hmri_enriched.csv` + `data/openalex_cache.json` + `data/openalex_coverage_report.json`

**Review coverage report:**
```bash
cat data/openalex_coverage_report.json
```

This shows what % of papers have complete data. Papers with missing required metrics will be excluded from scoring.

---

### 2. Edit themes.yaml

**Option A: Simple themes (flat structure):**

```yaml
themes:
  - id: dementia_intervention
    name: "Dementia Awareness & Intervention"
    narrative: |
      Research focused on Alzheimer's disease, dementia, brain degeneration...
    semantic_threshold: 0.5
    max_candidates: 100
    max_results: 50
    min_llm_relevance: 6.0
```

**Option B: Topics within themes (hierarchical structure):**

If you have **3 themes with 5 topics each** that need individual scoring and ranking:

```yaml
themes:
  # Theme 1, Topic 1
  - id: env_health_1_housing_food_water
    name: "Housing, Food, and Water Access"
    parent_theme: "Environmental Health and Disasters"
    theme_number: 1
    topic_number: 1
    narrative: |
      Research on housing quality, food security, water access...
    semantic_threshold: 0.5
    max_candidates: 100
    max_results: 50
    min_llm_relevance: 6.0

  # Theme 1, Topic 2-5...
  # Theme 2, Topics 1-5...
  # Theme 3, Topics 1-5...
```

See `themes_example_hierarchical.yaml` for complete example and `docs/hierarchical_themes_guide.md` for guidance.

---

### 3. Run theme search

```bash
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/hmri_themes_2025/
```

**Or if you skipped enrichment:**
```bash
python scripts/run_theme_search.py \
  --dataset "HMRI Pub Abstracts_20250703.csv" \
  --themes themes.yaml \
  --output results/hmri_themes_2025/
```

**What happens:**
1. Loads CSV and deduplicates → ~30K-40K unique publications
2. Checks data completeness for each paper
3. Uses existing `.embeddings_cache/` → **no embedding costs!**
4. For each theme (or topic if hierarchical):
   - Semantic search across all papers
   - GPT-4o-mini analyzes top candidates (uses `.llm_relevance_cache/`)
   - Filters by relevance threshold
   - Calculates SCImago-style scores (only for papers with complete data)
   - Papers with missing data are flagged but still included in outputs
   - Aggregates by staff (using `all_staff_ids` arrays)

**Processing time:**
- Flat structure (3 themes): ~5-10 minutes
- Hierarchical structure (3 themes × 5 topics = 15 searches): ~15-25 minutes

**Data Quality Handling:**
- Papers with ANY missing required metrics are excluded from theme score calculations
- Excluded papers are still listed in outputs with `excluded_from_scoring: True` flag
- Coverage statistics shown in theme_comparison.csv

**Cost:** Only new GPT analyses (~$1-2 per theme/topic if cache misses)

---

### 4. Analyze hierarchical results (if using Option B)

If you used hierarchical structure, rank topics within each parent theme:

```bash
python scripts/analyze_hierarchical_themes.py \
  --results results/hmri_themes_2025/theme_comparison.csv \
  --output results/hmri_themes_2025/
```

**What it produces:**
- `topic_rankings_by_theme.csv` - Topics ranked 1-5 within each parent theme
- `parent_theme_summary.csv` - Summary comparison of 3 parent themes
- `hierarchical_theme_report.md` - Detailed report with rankings and insights

**Processing time:** <1 minute

**See:** `docs/hierarchical_themes_guide.md` for complete documentation

---

## Output Files

**Flat structure:**
```
results/hmri_themes_2025/
├── theme_comparison.csv              # Cross-theme rankings
│
├── dementia_intervention/
│   ├── papers.csv                    # Unique papers for this theme
│   └── staff_summary.csv             # Aggregated by NumberPlate
│
└── [other themes]/
```

**Hierarchical structure (with Step 4 analysis):**
```
results/hmri_themes_2025/
├── theme_comparison.csv                  # All 15 topics (flat list)
├── topic_rankings_by_theme.csv           # Topics ranked within parent themes
├── parent_theme_summary.csv              # Summary of 3 parent themes
├── hierarchical_theme_report.md          # Detailed report
│
├── env_health_1_housing_food_water/      # Topic 1 of Theme 1
│   ├── papers.csv
│   └── staff_summary.csv
│
├── env_health_2_climate_health/          # Topic 2 of Theme 1
│   ├── papers.csv
│   └── staff_summary.csv
│
└── [13 more topic folders]/
```

### theme_comparison.csv

| Column | Description |
|--------|-------------|
| publications | # unique papers in theme |
| publications_scored | # papers with complete data used for scoring |
| publications_excluded | # papers excluded due to missing data |
| data_completeness_rate | % papers with complete data |
| theme_score | Overall score 0-100 (from scored papers only) |
| research_score | Research performance 0-100 |
| societal_score | Altmetric coverage (% of papers with attention) |
| unique_staff | # distinct NumberPlates |
| q1_percentage | % in Q1 journals |
| normalized_impact | Citations vs field average |
| excellence_rate | % in top 10% most cited |
| altmetric_coverage | % papers with social attention |
| avg_altmetric_score | Mean Altmetric score for attended papers |

### papers.csv (per theme)

Each row = **one unique publication** with:
- All original HMRI fields
- OpenAlex enrichment fields (if enriched)
- `all_staff_ids` = array of all NumberPlates
- `llm_relevance_score` (0-10) = GPT assessment
- `llm_reasoning` = explanation of fit
- `hybrid_score` = combined similarity score
- `excluded_from_scoring` = True if missing required data
- `missing_metrics` = list of missing required fields
- `data_quality_score` = % of required metrics present (0-1)

### staff_summary.csv (per theme)

Aggregated by NumberPlate:
- `paper_count` = publications in this theme
- `avg_relevance` = average LLM score
- `high_quality_papers` = count with score ≥ 8.0

---

## Example: Year-Filtered Analysis

To analyze only recent publications:

```bash
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/hmri_themes_2020_2024/ \
  --min-year 2020 \
  --max-year 2024
```

## Understanding Data Quality

After running theme search, check the coverage in `theme_comparison.csv`:

```csv
theme_id,publications,publications_scored,publications_excluded,data_completeness_rate
dementia,112,98,14,87.5%
```

**What this means:**
- 112 papers matched the theme
- 98 had complete data and contributed to the theme score
- 14 were excluded due to missing metrics
- 87.5% completeness rate

**If completeness is low (<70%):**
1. Review `papers.csv` → check `missing_metrics` column
2. Consider running OpenAlex enrichment if you haven't
3. Review `data/openalex_coverage_report.json` to see which metrics are commonly missing
4. Decide whether to implement imputation strategies (future enhancement)

---

## Adjusting Parameters

If initial results are off, edit `themes.yaml`:

```yaml
# Too many irrelevant papers? Increase:
semantic_threshold: 0.6        # Stricter embedding match
min_llm_relevance: 7.0         # Only highly relevant

# Too few results? Decrease:
semantic_threshold: 0.45       # Broader semantic net
min_llm_relevance: 5.5         # Include tangentially related

# More GPT analysis? Increase:
max_candidates: 150            # Analyze more papers
```

Then re-run step 3. **Caches ensure no recomputation of existing work!**

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

**Note:** `calculate_baselines.py` script exists for alternative field normalization approaches, but is NOT part of the default workflow. May be used later for imputation strategies.

## Troubleshooting

### "High exclusion rate (>30%)"
- Most likely: missing field-normalized impact data
- Solution: Run OpenAlex enrichment to fill gaps
- Review `missing_metrics` column in papers.csv
- Consider imputation strategy after understanding patterns

### "No results for theme X"
- Threshold too high or narrative too specific
- Try `semantic_threshold: 0.45`
- Check if abstracts are populated in CSV

### "Module not found" errors
```bash
# Make sure you're in the project directory
cd /Users/chris/Projects/bibtex-analyser-llm

# Install dependencies if needed
pip install pandas numpy pyyaml openai scikit-learn requests
```

### "Deduplication seems wrong"
- Check `Publication_ID` column exists and is populated
- Deduplication uses DOI first, then normalized title
- All NumberPlate values preserved in `all_staff_ids` array

### Check cache usage
```bash
# See how many embeddings cached
ls .embeddings_cache/*.pkl | wc -l

# See how many LLM analyses cached
ls .llm_relevance_cache/*.json | wc -l

# See OpenAlex cache
ls -lh data/openalex_cache.json
```

---

## Data Notes

**Column mappings (automatic):**
- `Output_Title` → `title`
- `Output_abstract` → `abstract`
- `NumberPlate` → `staff_id` (collected into `all_staff_ids`)
- `Publication_ID` → `ID` (used for deduplication)
- `Reported_Year` → `year`

**Deduplication logic:**
1. Group by `Publication_ID` first (exact match)
2. If no ID, group by DOI
3. If no DOI, group by normalized title
4. All `NumberPlate` values preserved in `all_staff_ids` array

---

## Next Steps

1. ✅ Run baseline calculation
2. ✅ Edit `themes.yaml` with your strategic themes
3. ✅ Run theme search
4. 📊 Review `theme_comparison.csv` for rankings
5. 🔍 Examine individual theme folders for details
6. 👥 Check `staff_summary.csv` files for researcher contributions

**Questions?** See `scripts/README_THEME_ANALYSIS.md` for detailed docs.
