# Data Quality Guide for Theme Analysis

## Overview

This guide explains the data quality approach used in theme analysis scoring, how to interpret data quality reports, and strategies for improving completeness rates.

---

## Core Philosophy: All-or-Nothing Scoring

### The Approach

Papers contribute to theme scores **only if they have complete data** for all required metrics. This ensures:

✅ **Internal consistency** - All scored papers measured by same criteria
✅ **Comparability** - Theme scores based on apples-to-apples comparisons
✅ **Transparency** - Explicit tracking of what's included/excluded
✅ **Informed decisions** - Can assess impact of missing data before deciding on fixes

### Phase 1: Assess

**Current strategy (all-or-nothing):**
1. Run theme analysis with existing + OpenAlex data
2. Review data completeness rates per theme
3. Identify patterns in missing metrics
4. Understand which papers are being excluded and why

### Phase 2: Impute (if needed)

**Future strategy (only if exclusion rate >30%):**
1. Analyze missing data patterns from Phase 1 outputs
2. Decide on appropriate imputation methods per metric
3. Implement imputation with clear documentation
4. Re-run analysis and compare results

---

## Required Metrics for Scoring

Papers need **ALL 7** of these metrics to contribute to theme scores:

### 1. Field-Normalized Impact
**Primary source:** `openalex_fwci_approx`
**Why required:** Fair comparison across research fields and publication ages
**How to get:** Run `scripts/enrich_with_openalex.py` to fetch via DOI lookup
**Typical issue:** Papers without DOIs or not indexed in OpenAlex

### 2. Journal Quality
**Primary sources:** `Clarivate_Quartile_Rank` OR `SJR_Best_Quartile`
**Why required:** Indicator of publication venue prestige (Q1 = top quartile)
**How to get:** Usually in institutional CSV; OpenAlex doesn't provide this
**Typical issue:** Conference papers, books, or journals not ranked

### 3. Citations
**Primary sources:** `Citations_Scopus` OR `Citations_WoS` OR `openalex_cited_by_count`
**Why required:** Core impact metric; needed even if FWCI available
**How to get:** Usually in institutional CSV; OpenAlex provides backup
**Typical issue:** Very recent papers (current year) may have zero citations

### 4. Open Access
**Primary sources:** `Ref_PMC_ID` OR `openalex_is_oa`
**Why required:** SCImago scoring component (5% weight)
**How to get:** OpenAlex enrichment fills most gaps
**Typical issue:** Older papers published before OA movement

### 5. International Collaboration
**Primary sources:** `Countries` OR `openalex_is_international`
**Why required:** SCImago scoring component (15% weight)
**How to get:** Usually in institutional CSV; OpenAlex provides backup
**Typical issue:** Single-author papers or missing affiliation data

### 6. Altmetrics
**Primary source:** `Altmetrics_Score`
**Why required:** Societal impact component (35% of theme score)
**How to get:** Usually in institutional CSV; no alternative source currently
**Typical issue:** Older papers or niche publications may lack social media attention

### 7. Author Position
**Primary sources:** `First_Author`, `Last_Author`
**Why required:** Leadership rate component (10% weight)
**How to get:** Usually in institutional CSV
**Typical issue:** Missing authorship metadata or single-author papers

---

## Interpreting Data Quality Reports

### theme_comparison.csv

Each theme row shows overall data quality:

```csv
theme_id,publications,publications_scored,publications_excluded,data_completeness_rate,theme_score
dementia,112,98,14,87.5,72.1
housing,87,62,25,71.3,65.8
```

**Key columns:**
- `publications`: Total papers matching theme (via semantic + LLM search)
- `publications_scored`: Papers with complete data used in scoring
- `publications_excluded`: Papers missing one or more required metrics
- `data_completeness_rate`: Percentage with complete data (scored / total × 100)

**Interpretation guide:**
- **≥90% completeness:** Excellent - theme score highly reliable
- **80-89% completeness:** Good - minimal bias from exclusions
- **70-79% completeness:** Acceptable - review excluded papers for systematic patterns
- **<70% completeness:** Concerning - investigate missing data and consider imputation

### [theme]/papers.csv

Each paper shows detailed quality information:

```csv
ID,title,llm_relevance_score,excluded_from_scoring,missing_metrics,data_quality_score
123,"Paper A",8.5,False,[],1.0
124,"Paper B",7.2,True,"['field_normalized_impact','open_access']",0.71
125,"Paper C",9.1,True,"['altmetrics']",0.86
```

**Key columns:**
- `excluded_from_scoring`: Boolean - True if paper missing any required metric
- `missing_metrics`: List of missing metric names (empty if complete)
- `data_quality_score`: Proportion of required metrics present (0-1 scale)

**Use cases:**
- **Identify patterns:** Sort by `missing_metrics` to see which metrics commonly missing
- **Assess bias:** Filter excluded papers - are they older? From specific fields?
- **Prioritize fixes:** Papers with high `llm_relevance_score` but excluded are high priority

### data/openalex_coverage_report.json

Generated by OpenAlex enrichment script:

```json
{
  "total_papers": 35000,
  "papers_with_doi": 32000,
  "openalex_found": 28000,
  "openalex_not_found": 4000,
  "coverage_rate": 87.5,
  "metrics_filled": {
    "openalex_fwci_approx": 27500,
    "openalex_is_oa": 28000,
    "openalex_is_international": 27800
  }
}
```

**Interpretation:**
- If `coverage_rate < 80%`, check DOI quality in source data
- `metrics_filled` shows how many papers got each enrichment field
- Papers in `openalex_not_found` may be too recent or not scholarly works

---

## Common Missing Data Patterns

### Pattern 1: Recent Publications (Last 6 Months)

**Symptoms:**
- High exclusion rate for papers from current year
- Missing: Citations (zero), Altmetrics (none yet), FWCI (insufficient time)

**Solutions:**
- **Accept exclusion:** Recent papers naturally lack citation/impact data
- **Temporal filtering:** Use `--min-year 2020 --max-year 2024` to exclude current year
- **Imputation (Phase 2):** Assign neutral scores (FWCI=1.0, citations=0) with flag

### Pattern 2: Non-Journal Publications

**Symptoms:**
- Conference papers, book chapters excluded
- Missing: Journal quartile rank (not applicable)

**Solutions:**
- **Document scope:** Note that theme scores focus on journal articles
- **Separate analysis:** Create conference-paper-specific themes with adapted scoring
- **Imputation (Phase 2):** Assign equivalent quartile based on conference ranking

### Pattern 3: Pre-Digital Era Publications

**Symptoms:**
- Papers before 2000 excluded
- Missing: DOI, OpenAlex data, Altmetrics

**Solutions:**
- **Accept limitation:** Pre-2000 papers lack modern metrics
- **Temporal filtering:** Focus analysis on 2000+ or 2010+
- **Manual enrichment:** Add DOIs for key older papers if critical to theme

### Pattern 4: Niche/Regional Research

**Symptoms:**
- Papers in non-English journals or regional topics excluded
- Missing: Clarivate/SJR quartile, OpenAlex indexing, Altmetrics

**Solutions:**
- **Broaden sources:** Use regional journal ranking systems
- **Document limitation:** Note that scoring favors internationally indexed research
- **Alternative metrics:** Consider field-specific impact measures

### Pattern 5: Altmetrics Gaps

**Symptoms:**
- 30-50% papers missing altmetrics
- Randomly distributed across years and fields

**Solutions:**
- **Accept reality:** Not all research gets social media attention
- **Adjust weights:** Consider reducing societal impact from 35% to 20%
- **Binary approach:** Presence/absence rather than score magnitude
- **Imputation (Phase 2):** Assign median altmetric score by field/year

---

## Improving Data Completeness

### Quick Wins (High Impact, Low Effort)

#### 1. Run OpenAlex Enrichment

```bash
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com
```

**Expected improvement:** +10-20% completeness for FWCI, OA status, collaboration

#### 2. Fix Missing DOIs

Check for patterns in papers without DOIs:

```python
import pandas as pd
df = pd.read_csv('HMRI Pub Abstracts_20250703.csv')
no_doi = df[df['Ref_DOI'].isna()]
print(no_doi[['Output_Title', 'Reported_Year', 'Publication_Type']].head(20))
```

**Solutions:**
- Crossref DOI lookup by title/author
- Manual DOI assignment for high-impact papers
- Journal website lookup for recent publications

#### 3. Fill Authorship Flags

Check if `First_Author` and `Last_Author` can be derived from `Author_Position` and `Pub_Max_Position`:

```python
# If Author_Position == 1 → First_Author = True
# If Author_Position == Pub_Max_Position → Last_Author = True
```

**Expected improvement:** +5-15% completeness

### Medium Effort

#### 4. Add Missing Journal Quartiles

For papers with journal names but missing quartile:
- Lookup in Scopus SJR database (free access)
- Use journal FoR codes to estimate quartile
- Match against institutional journal lists

#### 5. Backfill Altmetrics

Use Altmetric API (limited free tier):
- Fetch altmetric scores for papers with DOIs
- Cache responses to avoid repeated lookups
- Focus on high-relevance papers first

### Advanced (Phase 2 Only)

#### 6. Predictive Imputation

Build models to estimate missing values:

**FWCI prediction:**
```python
# Train on papers with FWCI
# Features: FoR code, publication type, journal quartile, year
# Predict FWCI for papers missing OpenAlex data
```

**Quartile prediction:**
```python
# Use journal title/ISSN to lookup in external databases
# Use FoR code and institutional reputation as fallback
```

#### 7. Alternative Baselines

Use `scripts/calculate_baselines.py` to create dataset-specific normalization:

```bash
python scripts/calculate_baselines.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_baselines.json
```

Replace OpenAlex FWCI with institution-specific field normalization.

**Pros:** Works for papers not in OpenAlex
**Cons:** Less accurate than global benchmarks; requires large dataset per field

---

## Decision Framework: When to Impute

### Don't Impute If:

❌ **Completeness >80%** - Exclusions unlikely to bias results significantly
❌ **Missing data is random** - No systematic patterns in exclusions
❌ **First analysis** - Always assess patterns before imputing
❌ **Imputation would introduce more uncertainty than exclusion**

### Consider Imputation If:

✅ **Completeness <70%** - Large proportion of papers excluded
✅ **Systematic bias** - Excluded papers differ systematically (e.g., all conference papers, all pre-2010)
✅ **Critical themes affected** - Top institutional priority themes have low completeness
✅ **Good predictive features available** - Can impute with reasonable accuracy

### Imputation Best Practices

1. **Document everything** - Record which papers imputed, method used, confidence level
2. **Sensitivity analysis** - Run scoring with/without imputed values, compare results
3. **Flag imputed papers** - Add `imputed_metrics: ['fwci', 'quartile']` column
4. **Report separately** - Show theme scores for: (a) complete data only, (b) complete + imputed
5. **Version control** - Save outputs before and after imputation for comparison

---

## Example Workflow

### Step 1: Initial Assessment

```bash
# Run theme analysis with OpenAlex enrichment
python scripts/enrich_with_openalex.py --input data.csv --output data_enriched.csv
python scripts/run_theme_search.py --dataset data_enriched.csv --themes themes.yaml --output results/

# Check completeness
grep -h "data_completeness_rate" results/theme_comparison.csv
```

### Step 2: Investigate Low Completeness

```bash
# Find themes with <70% completeness
awk -F',' 'NR>1 && $5<70 {print $1, $5}' results/theme_comparison.csv

# For each low-completeness theme, check missing patterns
cd results/dementia_intervention/
awk -F',' 'NR>1 && $6=="True" {print $7}' papers.csv | sort | uniq -c | sort -rn
```

**Example output:**
```
  45 ['field_normalized_impact']
  23 ['altmetrics']
  18 ['field_normalized_impact', 'open_access']
  12 ['journal_quality']
```

**Interpretation:** Field-normalized impact is the main gap (45 papers missing FWCI)

### Step 3: Targeted Fixes

```bash
# Check which papers missing FWCI
awk -F',' 'NR>1 && $7~"field_normalized_impact" {print $1,$2,$3}' papers.csv > missing_fwci.csv

# Investigate: Are they missing DOIs? Not in OpenAlex? Conference papers?
```

### Step 4: Decide and Document

Create `results/data_quality_report.md`:

```markdown
# Data Quality Report - Theme Analysis 2025

## Summary
- Overall completeness: 78.5%
- Themes affected: Dementia (71%), Housing (82%), Indigenous Health (65%)

## Key Findings
1. **FWCI gaps (45 papers)**: Papers without DOIs → not enriched by OpenAlex
2. **Altmetrics gaps (23 papers)**: Older papers (pre-2015) → low social media coverage
3. **Quartile gaps (12 papers)**: Conference papers → no journal quartile

## Actions Taken
1. ✅ OpenAlex enrichment completed
2. ✅ Manual DOI assignment for 10 high-impact papers → +10 papers scored
3. ⏭️ Altmetrics imputation deferred - pattern is random, not systematic

## Recommendations
- Accept current 78.5% completeness for Phase 1 reporting
- Reassess after next annual data update
- Consider conference-paper-specific themes for excluded research
```

---

## Tools and Scripts

### Check Data Quality

```python
# Quick check of data completeness
import pandas as pd

comparison = pd.read_csv('results/theme_comparison.csv')
print("Themes by completeness:\n")
print(comparison[['theme_id', 'data_completeness_rate', 'publications_excluded']].sort_values('data_completeness_rate'))

# Identify most common missing metrics across all themes
import os
missing_counts = {}
for theme_dir in os.listdir('results'):
    papers_file = f'results/{theme_dir}/papers.csv'
    if os.path.exists(papers_file):
        df = pd.read_csv(papers_file)
        for metrics in df['missing_metrics'].dropna():
            for metric in eval(metrics):  # Convert string list to actual list
                missing_counts[metric] = missing_counts.get(metric, 0) + 1

print("\nMost common missing metrics:")
for metric, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {metric}: {count}")
```

### Generate Quality Report

```bash
# Create comprehensive quality report
cat > generate_quality_report.sh <<'EOF'
#!/bin/bash
echo "# Data Quality Report - Generated $(date)" > quality_report.md
echo "" >> quality_report.md

echo "## Overall Statistics" >> quality_report.md
awk -F',' 'NR>1 {total+=$2; scored+=$3; excluded+=$4} END {
  print "- Total publications across themes:", total
  print "- Publications scored:", scored
  print "- Publications excluded:", excluded
  print "- Overall completeness:", (scored/total*100) "%"
}' results/theme_comparison.csv >> quality_report.md

echo "" >> quality_report.md
echo "## Per-Theme Completeness" >> quality_report.md
awk -F',' 'NR>1 {print "- " $1 ":", $5 "% (" $4 " excluded)"}' results/theme_comparison.csv >> quality_report.md
EOF

chmod +x generate_quality_report.sh
./generate_quality_report.sh
```

---

## FAQ

### Q: Why not just impute missing values from the start?

**A:** Imputation can introduce bias and hide systematic data quality issues. By running all-or-nothing first, we:
- See the true extent of missing data
- Identify patterns that might indicate systematic issues
- Make informed decisions about whether imputation is needed
- Maintain transparency about what's included in scores

### Q: What if a high-priority theme has low completeness?

**A:**
1. Run targeted data enrichment for that theme's papers
2. Manually add missing data for high-relevance publications (LLM score ≥8)
3. Document limitations in reporting
4. Consider alternative metrics or adapted scoring for that theme

### Q: Can I change the required metrics?

**A:** Yes, but carefully:
- Edit `theme_analysis/scoring.py` to modify required fields
- Update documentation to reflect changes
- Ensure all scoring components still have necessary data
- Consider creating separate scoring profiles (strict vs. lenient)

### Q: How do I handle zero values vs. missing values?

**A:**
- **Zero citations:** Valid data point (paper genuinely has zero citations) → include
- **Missing citation field:** Unknown citation count → exclude
- **Zero altmetric score:** No social media attention (valid) → include
- **Missing altmetric field:** No altmetric data available → exclude

Use `pd.isna()` to distinguish missing from zero.

### Q: Should I exclude current-year publications?

**A:** Recommended for citation-based metrics:

```bash
python scripts/run_theme_search.py \
  --dataset data.csv \
  --themes themes.yaml \
  --max-year 2024 \  # Exclude 2025 publications
  --output results/
```

This avoids excluding recent papers simply because they haven't had time to accumulate citations.

---

## Summary

✅ **Phase 1 (current):** All-or-nothing approach with explicit exclusion tracking
✅ **Assess first:** Understand patterns before implementing fixes
✅ **OpenAlex enrichment:** Quick win for filling most common gaps
✅ **Document everything:** Transparency about what's included/excluded
⏭️ **Phase 2 (if needed):** Targeted imputation with clear documentation

The goal is **reliable, interpretable theme scores** - not perfect data coverage. By being transparent about data quality, we enable informed interpretation of results.
