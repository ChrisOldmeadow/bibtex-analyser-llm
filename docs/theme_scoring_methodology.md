# Theme Scoring Methodology

## Overview

This document describes the institutional-style scoring system for ranking publications within strategic research themes, adapted from the SCImago Institutions Rankings methodology. Each theme receives an overall score (0-100) based on research performance and societal impact metrics.

---

## Data Completeness Requirements

### Required Metrics for Scoring

Papers must have **ALL** of the following metrics to contribute to theme scores:

1. **Field-normalized impact** - OpenAlex FWCI approximation (`openalex_fwci_approx`)
2. **Journal quality** - `Clarivate_Quartile_Rank` OR `SJR_Best_Quartile`
3. **Citations** - `Citations_Scopus` OR `Citations_WoS` OR `openalex_cited_by_count`
4. **Open access** - `Ref_PMC_ID` OR `openalex_is_oa`
5. **International collaboration** - `Countries` OR `openalex_is_international`
6. **Altmetrics** - `Altmetrics_Score`
7. **Author position** - `First_Author`, `Last_Author`

### Exclusion Policy

**All-or-nothing approach:**
- Papers missing **ANY** required metric are excluded from theme score calculations
- Excluded papers still appear in outputs with flags:
  - `excluded_from_scoring: True`
  - `missing_metrics: ['field_normalized_impact', 'open_access']`
  - `data_quality_score: 0.71` (5/7 metrics present)

### Data Sources

**Primary source (HMRI CSV):**
- Citations, journal quality, authorship, countries, altmetrics, FoR codes
- May have gaps in some publications

**Enrichment source (OpenAlex):**
- Field-normalized citation impact (FWCI proxy via percentile)
- Open access status and type (gold/green/bronze)
- International collaboration flags
- Additional citation counts for validation
- Fetched via DOI lookup using `scripts/enrich_with_openalex.py`

### Coverage Reporting

All outputs include data quality metrics:

**In `theme_comparison.csv`:**
- `publications`: Total papers matching theme
- `publications_scored`: Papers with complete data (used in scoring)
- `publications_excluded`: Papers excluded due to missing data
- `data_completeness_rate`: % papers with complete required metrics

**In `[theme]/papers.csv`:**
- `excluded_from_scoring`: Boolean flag
- `missing_metrics`: List of missing required fields
- `data_quality_score`: Proportion of required metrics present (0-1)

### Imputation Strategy

**Phase 1 (current):** All-or-nothing exclusion
- Run pipeline to assess data completeness rates
- Review `missing_metrics` column to identify patterns
- Check OpenAlex coverage report for enrichment success rates

**Phase 2 (future, if needed):**
- If exclusion rate >30%, consider imputation strategies
- Options: median imputation, predictive models, alternative data sources
- Decision made after understanding missing data patterns

---

## Available Publication Data Fields

Based on the institutional publication export, the following fields are available for scoring:

### Citation Metrics
- `Citations_Scopus`: Scopus citation count
- `Citations_WoS`: Web of Science citation count
- `Citations_Europe_PubMed`: PubMed Europe citations
- `Citations_WoS_Lite`: Web of Science Lite citations
- `Citations_SciVal`: SciVal citation metrics
- `openalex_cited_by_count`: OpenAlex citation count (from enrichment)
- `openalex_fwci_approx`: Field-Weighted Citation Impact proxy (from enrichment)
- `openalex_citation_percentile`: Citation percentile within publication year (from enrichment)

### Journal Quality Indicators
- `Clarivate_Quartile_Rank`: Journal quartile (Q1, Q2, Q3, Q4)
- `SJR_Best_Quartile`: SCImago Journal Rank quartile
- `Journal_FoR_Codes`: Field of Research codes for journal

### Authorship & Collaboration
- `Author_Position`: Position in author list
- `Pub_Max_Position`: Total number of authors
- `First_Author`: Boolean flag for first authorship
- `Last_Author`: Boolean flag for last/corresponding authorship
- `Countries`: List of countries involved (for international collaboration)
- `Total_Authors`: Total author count
- `UoN_Authors`: University of Newcastle author count
- `openalex_is_international`: International collaboration flag (from enrichment)
- `openalex_countries`: Country codes from OpenAlex (from enrichment)

### Impact & Visibility
- `Altmetrics_Score`: Altmetric attention score
- `Altmetrics_Total_Posts`: Total social media mentions

### Open Access
- `Ref_PMC_ID`: PubMed Central ID (definitive OA indicator)
- `openalex_is_oa`: Open access status (from enrichment)
- `openalex_oa_status`: OA type - gold, green, bronze, closed (from enrichment)

### Research Classification
- `FoR_Codes_2020`: Field of Research codes (2020)
- `FoR_Codes_2020_Desc`: FoR descriptions
- `SciVal_SDGs`: Sustainable Development Goals (SciVal)
- `SDGs`: SDG classifications
- `openalex_concepts`: Top 3 research concepts (from enrichment)

### Publication Metadata
- `HERDC_Category`: Australian research category (A, B, C, etc.)
- `Publication_Type`: Article, Review, Conference, etc.
- `Reported_Year`: Publication year
- `openalex_type`: Publication type from OpenAlex (from enrichment)

---

## Scoring Algorithm

The theme score is calculated using a **two-stage approach** similar to SCImago Institutions Rankings:

### Stage 1: Research Performance Score (0-100)

Weighted combination of research quality indicators:

```
Research Score = (
    0.15 × Output_Normalized +           # Quantity of publications
    0.15 × Intl_Collaboration_Rate +     # International collaboration
    0.20 × Q1_Percentage +               # Top-quartile publications
    0.20 × Normalized_Impact +           # Field-normalized citation impact
    0.15 × Excellence_Rate +             # Highly-cited publications
    0.10 × Leadership_Rate +             # First/last authorship
    0.05 × Open_Access_Rate              # Open access availability
) × 100
```

#### Component Definitions

**1. Output_Normalized (25%)**
- Number of publications matching the theme **with complete data**
- Normalized: `min(publication_count / output_cap, 1.0)` where `output_cap` is the 90th percentile of publication counts across all themes (fallback: 250)
- Rationale: Keeps the metric informative even when one theme is an outlier; large themes above the percentile earn full credit
- Note: Only counts `publications_scored`, not `publications_excluded`

**2. International Collaboration Rate (15%)**
- Percentage of publications with authors from multiple countries
- **Data sources (priority order):**
  1. `openalex_is_international` (from enrichment) - direct boolean flag
  2. `Countries` field (from HMRI CSV) - split by semicolon/comma, count unique
- Formula: `(papers_with_intl_collab / total_scored_papers) × 100`
- Papers missing both fields are excluded from scoring

**3. Q1 Percentage (20%)**
- Percentage of publications in top-quartile journals
- Uses `Clarivate_Quartile_Rank` or `SJR_Best_Quartile`
- Priority: Clarivate > SJR (use best available)
- Formula: `(q1_papers / total_scored_papers) × 100`
- Papers missing both quartile fields are excluded from scoring

**4. Normalized Impact (10%)**
- Field-normalized citation rate relative to world average
- **Primary source:** `openalex_fwci_approx` (from OpenAlex enrichment)
  - Calculated from citation percentile within publication year
  - Values: 0.5 = below average, 1.0 = world average, 2.0+ = twice average
  - Already field-normalized by OpenAlex's methodology
- **Calculation:**
  ```python
  # Use FWCI approximation directly from OpenAlex
  fwci = paper.openalex_fwci_approx

  # Average across all papers in theme
  avg_fwci = mean([fwci for each scored paper])

  # For scoring, contribution = min(avg_fwci / 2.0, 1.0)
  ```
- Papers missing `openalex_fwci_approx` are excluded from scoring

**5. Excellence Rate (15%)**
- Percentage of publications with exceptionally high field-normalized impact
- **Calculation:**
  ```python
  # Count papers with FWCI >= 2.0 (twice world average)
  # This approximates "top 10% most cited" threshold

  excellence_count = sum(1 for paper in theme
                        if paper.openalex_fwci_approx >= 2.0)

  excellence_rate = (excellence_count / total_scored_papers) × 100
  ```
- Alternative: Use `openalex_citation_percentile >= 90` if percentile data available
- Papers missing FWCI data are excluded from scoring

**6. Leadership Rate (10%)**
- Percentage of papers where institution has first or last/corresponding author
- Uses `First_Author` and `Last_Author` boolean flags
- Formula: `(leadership_papers / total_scored_papers) × 100`
- Papers missing both authorship flags are excluded from scoring

**7. Open Access Rate (5%)**
- Percentage of publications freely accessible
- **Data sources (priority order):**
  1. `openalex_is_oa` (from enrichment) - boolean flag
  2. `Ref_PMC_ID` (from HMRI CSV) - presence indicates OA
- Formula: `(oa_papers / total_scored_papers) × 100`
- Papers missing both OA indicators are excluded from scoring

---

### Stage 2: Societal Impact Score (0-100)

Measures the share of publications that attracted measurable online attention:

```
Societal Score = Altmetric_Coverage_Rate × 100
```

#### Component Definition

- **Altmetric Coverage Rate**
  - Percentage of scored publications with `Altmetrics_Score > 0`
  - Uses `Altmetrics_Score` (or `altmetrics_score`) from the HMRI CSV
  - Formula: `(papers_with_altmetrics / total_scored_papers)`
  - Papers missing altmetric data are excluded from both numerator and denominator
  - The raw `avg_altmetric_score` is reported separately for context but does not influence the score

---

### Stage 3: Final Theme Score (0-100)

Weighted combination of the two component scores:

```
Theme Score = (
    0.65 × Research_Score +      # Research performance (65%)
    0.35 × Societal_Score        # Societal impact (35%)
)
```

**Rationale for weights:**
- Research performance is primary focus (65%) for academic assessment
- Societal impact (35%) captures translation and engagement
- Innovation component from SCImago (typically 30%) is omitted due to lack of patent/industry data

**Important:** Theme score calculated only from papers with complete required data. Excluded papers do not contribute to any component scores.

---

## Handling Missing Data

### Detection

During theme search, each paper is checked for required metrics:

```python
def check_data_completeness(paper: Dict) -> Tuple[bool, List[str], float]:
    """Check if paper has all required metrics for scoring.

    Returns:
        (is_complete, missing_metrics, quality_score)
    """
    required = {
        'field_normalized_impact': paper.get('openalex_fwci_approx'),
        'journal_quality': paper.get('Clarivate_Quartile_Rank') or paper.get('SJR_Best_Quartile'),
        'citations': paper.get('Citations_Scopus') or paper.get('Citations_WoS') or paper.get('openalex_cited_by_count'),
        'open_access': paper.get('Ref_PMC_ID') or paper.get('openalex_is_oa'),
        'international_collab': paper.get('Countries') or paper.get('openalex_is_international'),
        'altmetrics': paper.get('Altmetrics_Score'),
        'authorship': paper.get('First_Author') is not None and paper.get('Last_Author') is not None
    }

    missing = [key for key, value in required.items() if not value]
    quality_score = (len(required) - len(missing)) / len(required)
    is_complete = len(missing) == 0

    return is_complete, missing, quality_score
```

### Reporting

**Per-theme outputs include:**

`papers.csv`:
```csv
ID,title,llm_relevance_score,hybrid_score,excluded_from_scoring,missing_metrics,data_quality_score
12345,"Paper Title",8.5,0.87,False,[],1.0
12346,"Another Paper",7.2,0.79,True,"['field_normalized_impact', 'open_access']",0.71
```

`theme_comparison.csv`:
```csv
theme_id,publications,publications_scored,publications_excluded,data_completeness_rate,theme_score
dementia,112,98,14,87.5,72.1
```

### Addressing Low Completeness

If `data_completeness_rate < 70%` for a theme:

1. **Review patterns:**
   - Check `missing_metrics` column in `papers.csv`
   - Identify which metrics are commonly missing
   - Review `data/openalex_coverage_report.json` if enrichment was used

2. **Improve coverage:**
   - Run OpenAlex enrichment if not already done
   - Check DOI quality in source data
   - Consider alternative data sources for specific metrics

3. **Assess bias:**
   - Are excluded papers systematically different? (older papers, specific fields)
   - Could exclusions skew theme scores?

4. **Consider imputation (Phase 2):**
   - Median/mode imputation for missing values
   - Predictive models based on available metrics
   - Alternative normalization methods (see `scripts/calculate_baselines.py`)

**Important:** Do not implement imputation until exclusion patterns are understood. All-or-nothing approach ensures internal consistency in Phase 1.

---

## Output Format

Theme comparison CSV will include:

```csv
theme_id,theme_name,publications,publications_scored,publications_excluded,data_completeness_rate,theme_score,research_score,societal_score,unique_staff,avg_relevance,high_quality_papers,q1_percentage,intl_collaboration_rate,normalized_impact,excellence_rate,leadership_rate,open_access_rate,altmetric_coverage,avg_altmetric_score
dementia_intervention,Dementia Awareness & Intervention,112,98,14,87.5,72.1,76.8,62.3,56,8.2,38,74.1,38.4,1.45,18.7,61.2,41.1,31.2,12.7
housing_health_equity,Healthy Housing & Community Resilience,87,79,8,90.8,67.3,71.2,59.8,42,7.8,23,68.4,42.5,1.23,12.6,58.6,34.5,24.1,8.3
```

This enables:
1. **Ranking themes** by overall score
2. **Assessing data quality** via completeness rates
3. **Identifying strengths** (which component scores are highest)
4. **Benchmarking** across institutional themes
5. **Tracking over time** (re-run annually to see score changes)
6. **Prioritizing data improvements** (focus on themes with low completeness)

---

## Advantages of This Approach

1. **Standardized**: Based on established SCImago methodology used globally
2. **Multi-dimensional**: Captures quantity, quality, collaboration, impact, and engagement
3. **Field-normalized**: Fair comparison across different research areas using OpenAlex FWCI
4. **Transparent**: All components are clearly weighted and documented
5. **Actionable**: Low scores in specific components suggest improvement strategies
6. **Data quality aware**: Explicit tracking and reporting of missing data
7. **Globally benchmarked**: Uses OpenAlex data for international comparisons
8. **Flexible**: Can adapt imputation strategies based on observed data patterns

---

## Implementation Notes

### OpenAlex Enrichment Integration

The `scripts/enrich_with_openalex.py` script fetches metrics before theme search:

```bash
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com
```

**What it adds:**
- `openalex_fwci_approx`: Field-Weighted Citation Impact (primary metric for normalized impact)
- `openalex_citation_percentile`: Year-based citation percentile
- `openalex_is_oa`: Open access flag
- `openalex_is_international`: International collaboration flag
- `openalex_cited_by_count`: Citation count for validation
- `openalex_concepts`: Top research concepts

**Cost:** Free API with polite rate limiting (10 requests/second)

**Time:** ~20-30 minutes first run (with caching), <1 minute subsequent runs

### Modified Scoring Function

The `theme_analysis/scoring.py` module implements SCImago scoring with data quality checks:

```python
def calculate_theme_score(theme_df: pd.DataFrame) -> Dict:
    """Calculate SCImago-style theme score from papers with complete data.

    Args:
        theme_df: DataFrame of publications (may include excluded papers)

    Returns:
        Dictionary with score components and data quality metrics
    """
    # Filter to only papers with complete data
    scored_papers = theme_df[~theme_df['excluded_from_scoring']]
    excluded_papers = theme_df[theme_df['excluded_from_scoring']]

    if scored_papers.empty:
        return {
            'publications': len(theme_df),
            'publications_scored': 0,
            'publications_excluded': len(excluded_papers),
            'data_completeness_rate': 0.0,
            'theme_score': 0,
            'research_score': 0,
            'societal_score': 0
        }

    # Calculate all components using only scored_papers
    # ... (implementation as shown in previous sections)

    return {
        'publications': len(theme_df),
        'publications_scored': len(scored_papers),
        'publications_excluded': len(excluded_papers),
        'data_completeness_rate': len(scored_papers) / len(theme_df) * 100,
        'theme_score': final_score,
        # ... other metrics
    }
```

### Alternative: Dataset Baselines (Optional)

The `scripts/calculate_baselines.py` script exists for alternative normalization approaches but is **NOT part of the default workflow**:

```bash
# Optional - only if OpenAlex enrichment insufficient
python scripts/calculate_baselines.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_baselines.json
```

**When to use:**
- OpenAlex enrichment doesn't cover enough papers
- Need institution-specific benchmarks
- Exploring alternative imputation strategies

**Not recommended for Phase 1** - focus on OpenAlex enrichment first.

---

## Next Steps

1. ✅ Implement OpenAlex enrichment script
2. ✅ Add data quality checks to pipeline
3. ✅ Update scoring function to use FWCI from OpenAlex
4. ✅ Add completeness tracking to all outputs
5. 🔄 Run pipeline on full dataset to assess exclusion rates
6. 📊 Review data quality reports and missing metric patterns
7. 🤔 Decide on Phase 2 imputation strategies if needed
8. 📈 Optional: Add visualization of data completeness per theme
