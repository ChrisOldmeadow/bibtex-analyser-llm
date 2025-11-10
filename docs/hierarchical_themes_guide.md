# Hierarchical Themes Guide

## Overview

This guide explains how to structure themes with **topics nested within parent themes**, score each topic independently, and then rank topics within their parent theme.

**Use case:** You have 3 strategic research themes, each with 5 specific topics that need individual scoring and ranking.

---

## Structure

```
Parent Theme 1: Environmental Health and Disasters
  ├─ Topic 1: Housing, food, and water access
  ├─ Topic 2: Climate change and health impacts
  ├─ Topic 3: Air quality and respiratory health
  ├─ Topic 4: Disaster preparedness and response
  └─ Topic 5: Environmental toxins and chronic disease

Parent Theme 2: [Your Second Theme]
  ├─ Topic 1: [Description]
  ├─ Topic 2: [Description]
  └─ ...

Parent Theme 3: [Your Third Theme]
  └─ ...
```

---

## Step-by-Step Workflow

### Step 1: Create themes.yaml with Hierarchical Structure

Use the naming convention: `{parent_theme_id}_{topic_number}_{topic_description}`

**Example:** See `themes_example_hierarchical.yaml`

Key fields for each topic:
```yaml
- id: env_health_1_housing_food_water          # Unique ID
  name: "Housing, Food, and Water Access"       # Display name
  parent_theme: "Environmental Health and Disasters"  # Parent theme name
  theme_number: 1                               # Parent theme number (1-3)
  topic_number: 1                               # Topic number within theme (1-5)
  narrative: |
    [Detailed description of what this topic covers...]
  semantic_threshold: 0.5
  max_candidates: 100
  max_results: 50
  min_llm_relevance: 6.0
```

**Important:** Include `parent_theme`, `theme_number`, and `topic_number` fields so the analysis script can group topics correctly.

### Step 2: Run Standard Theme Search

The theme search treats each topic as an independent search:

```bash
# Optional: Enrich with OpenAlex first
python scripts/enrich_with_openalex.py \
  --input "HMRI Pub Abstracts_20250703.csv" \
  --output data/hmri_enriched.csv \
  --cache data/openalex_cache.json \
  --email your.email@example.com

# Run theme search (will process all 15 topics = 3 themes × 5 topics)
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/hmri_themes_2025/
```

**What this produces:**
- Individual folder for each of the 15 topics with papers.csv and staff_summary.csv
- `theme_comparison.csv` with all 15 topics as separate rows

### Step 3: Analyze Hierarchical Results

Now use the hierarchical analysis script to group topics by parent theme:

```bash
python scripts/analyze_hierarchical_themes.py \
  --results results/hmri_themes_2025/theme_comparison.csv \
  --output results/hmri_themes_2025/
```

**What this produces:**

#### 1. `topic_rankings_by_theme.csv`

Topics ranked within each parent theme:

```csv
theme_id,theme_name,parent_theme,topic_number,rank_within_theme,theme_score,publications,publications_scored,data_completeness_rate,...
env_health_1_housing_food_water,Housing Food and Water Access,Environmental Health and Disasters,1,1,75.2,120,105,87.5,...
env_health_1_climate_health_impacts,Climate Change Health Impacts,Environmental Health and Disasters,2,2,72.8,98,89,90.8,...
env_health_1_air_pollution,Air Quality Respiratory Health,Environmental Health and Disasters,3,3,68.5,87,79,90.8,...
...
```

#### 2. `parent_theme_summary.csv`

Summary statistics for each parent theme:

```csv
rank,parent_theme,topics_count,publications_total,publications_scored,theme_score_avg,theme_score_max,theme_score_min,top_topic,top_topic_score
1,Environmental Health and Disasters,5,450,395,71.4,75.2,65.3,Housing Food and Water Access,75.2
2,[Second Theme],5,380,340,68.9,74.1,62.5,[Top topic name],74.1
3,[Third Theme],5,320,275,65.2,70.3,58.9,[Top topic name],70.3
```

#### 3. `hierarchical_theme_report.md`

Detailed markdown report with:
- Topic rankings table per parent theme
- Key insights (highest scoring, most publications, low completeness warnings)
- Cross-theme comparisons

---

## Output Interpretation

### Topic Rankings Within Theme

**Console output example:**
```
============================================================
PARENT THEME: Environmental Health and Disasters
============================================================

Topic Rankings (by theme_score):

  1. Housing, Food, and Water Access                      Score: 75.2 | Pubs: 120 (105 scored) | Completeness: 87.5%
  2. Climate Change and Health Impacts                    Score: 72.8 | Pubs:  98 ( 89 scored) | Completeness: 90.8%
  3. Air Quality and Respiratory Health                   Score: 68.5 | Pubs:  87 ( 79 scored) | Completeness: 90.8%
  4. Disaster Preparedness and Response                   Score: 67.1 | Pubs:  76 ( 65 scored) | Completeness: 85.5%
  5. Environmental Toxins and Chronic Disease             Score: 65.3 | Pubs:  69 ( 57 scored) | Completeness: 82.6%

Theme Summary:
  Total publications: 450
  Scored publications: 395
  Excluded publications: 55
  Average completeness: 87.4%
  Average theme score: 69.8
```

**Interpretation:**
- **Rank 1 topic** (Housing, Food, Water) is the strongest within this theme
- All topics have good completeness (>80%)
- Can prioritize resources toward top-ranked topics

### Parent Theme Comparison

```
============================================================
PARENT THEME COMPARISON
============================================================

1. Environmental Health and Disasters
   Average Score: 69.8 (range: 65.3-75.2)
   Topics: 5 | Publications: 450 (395 scored)
   Data Completeness: 87.4%
   Top Topic: Housing, Food, and Water Access (score: 75.2)

2. [Second Theme]
   Average Score: 68.9 (range: 62.5-74.1)
   ...
```

**Interpretation:**
- Compare strength across parent themes
- Identify which themes have highest-performing topics
- See data quality across themes

---

## Use Cases

### Use Case 1: Strategic Priority Setting

**Goal:** Identify top 3 topics across all themes for funding priority

**Approach:**
1. Review `topic_rankings_by_theme.csv` sorted by `theme_score` descending
2. Look at topics with `theme_score >= 70` and `publications_scored >= 50`
3. Check `unique_staff` to see research capacity

### Use Case 2: Within-Theme Resource Allocation

**Goal:** Understand topic distribution within Environmental Health theme

**Approach:**
1. Filter `topic_rankings_by_theme.csv` to `parent_theme == "Environmental Health and Disasters"`
2. Review `rank_within_theme` to see which topics are strongest
3. Compare `publications` counts to see where research is concentrated
4. Use `staff_summary.csv` files to identify key researchers per topic

### Use Case 3: Data Quality Improvement

**Goal:** Improve completeness for low-scoring topics

**Approach:**
1. Find topics with `data_completeness_rate < 80%` in `topic_rankings_by_theme.csv`
2. Review individual topic folders → check `papers.csv` → examine `missing_metrics` column
3. Prioritize enrichment for high-relevance papers (high `llm_relevance_score`)
4. Re-run theme search after data improvements

### Use Case 4: Year-over-Year Tracking

**Goal:** Track how topic rankings change annually

**Approach:**
```bash
# Run for 2024 data
python scripts/run_theme_search.py --dataset data_2024.csv --output results/2024/
python scripts/analyze_hierarchical_themes.py --results results/2024/theme_comparison.csv --output results/2024/

# Run for 2025 data
python scripts/run_theme_search.py --dataset data_2025.csv --output results/2025/
python scripts/analyze_hierarchical_themes.py --results results/2025/theme_comparison.csv --output results/2025/

# Compare
diff results/2024/topic_rankings_by_theme.csv results/2025/topic_rankings_by_theme.csv
```

---

## Tips and Best Practices

### Writing Topic Narratives

**Good practices:**
- **Be specific:** Describe exact research areas, not just keywords
- **Include examples:** List specific diseases, interventions, populations
- **Use synonyms:** Include alternative terms (e.g., "climate change" and "global warming")
- **Avoid overlap:** Make topics distinct enough that papers clearly belong to one topic

**Example - Too broad:**
```yaml
narrative: "Research on environmental health"
```

**Example - Good specificity:**
```yaml
narrative: |
  Research focused on protecting health and wellbeing through better access to
  affordable and reliable housing, healthy and affordable food, and clean water,
  particularly for vulnerable communities. This includes studies on:
  - Housing quality and health outcomes in disadvantaged populations
  - Food security, nutrition assistance programs, and food deserts
  - Water quality testing, contamination events, and access equity
  - Environmental determinants of health inequities
  - Community resilience to environmental stressors
```

### Adjusting Topic Parameters

If a topic returns too many/few results:

**Too many irrelevant papers:**
```yaml
semantic_threshold: 0.55    # Increase from 0.5
min_llm_relevance: 7.0      # Increase from 6.0
```

**Too few papers:**
```yaml
semantic_threshold: 0.45    # Decrease from 0.5
min_llm_relevance: 5.5      # Decrease from 6.0
max_candidates: 150         # Increase from 100
```

### Handling Topic Overlap

If papers appear in multiple topics:
- This is expected and normal
- Papers can legitimately contribute to multiple topics
- Individual topic folders will have duplicate papers across topics
- Staff aggregations will show researchers' contributions across topics

To analyze overlap:
```python
import pandas as pd
import os

# Load papers from two topics
topic1 = pd.read_csv('results/env_health_1_housing_food_water/papers.csv')
topic2 = pd.read_csv('results/env_health_1_climate_health_impacts/papers.csv')

# Find overlapping papers
overlap = set(topic1['ID']) & set(topic2['ID'])
print(f"Overlap: {len(overlap)} papers appear in both topics")
```

### Naming Conventions

Recommended ID format: `{parent}_{topic_num}_{short_desc}`

**Examples:**
- `env_health_1_housing` ✅
- `env_health_2_climate` ✅
- `indigenous_health_1_chronic_disease` ✅
- `envhealth1` ❌ (hard to parse)
- `topic_5` ❌ (no parent theme context)

---

## Complete Example: Environmental Health Theme

```yaml
themes:
  # Topic 1: Housing, Food, Water
  - id: env_health_1_housing_food_water
    name: "Housing, Food, and Water Access"
    parent_theme: "Environmental Health and Disasters"
    theme_number: 1
    topic_number: 1
    narrative: |
      Research focused on protecting health and wellbeing through better access to
      affordable and reliable housing, healthy and affordable food, and clean water,
      particularly for vulnerable communities.
    semantic_threshold: 0.5
    max_candidates: 100
    max_results: 50
    min_llm_relevance: 6.0

  # Topic 2: Climate Change
  - id: env_health_2_climate_health
    name: "Climate Change and Health Impacts"
    parent_theme: "Environmental Health and Disasters"
    theme_number: 1
    topic_number: 2
    narrative: |
      Research on the health impacts of climate change and extreme weather events,
      including heat-related illness, climate-sensitive diseases, and mental health
      impacts of climate disasters.
    semantic_threshold: 0.5
    max_candidates: 100
    max_results: 50
    min_llm_relevance: 6.0

  # ... continue for topics 3-5
```

**Run analysis:**
```bash
# Step 1: Theme search (analyzes all 5 topics independently)
python scripts/run_theme_search.py \
  --dataset data/hmri_enriched.csv \
  --themes themes.yaml \
  --output results/env_health_2025/

# Step 2: Hierarchical analysis (ranks topics within theme)
python scripts/analyze_hierarchical_themes.py \
  --results results/env_health_2025/theme_comparison.csv \
  --output results/env_health_2025/
```

**Result:** Topic rankings showing which of the 5 topics has strongest research performance.

---

## FAQ

### Q: Should topics be mutually exclusive?

**A:** Not necessarily. Some papers may legitimately fit multiple topics (e.g., a paper on "climate change impacts on food security" fits both climate and housing/food topics). This is fine - papers can contribute to multiple topic scores.

### Q: How do I handle topics with very different publication volumes?

**A:** The scoring already normalizes for volume:
- Output component normalizes to 100 publications = max score
- Other components are percentages (unaffected by volume)
- Small topics can score high if quality is high

If needed, you can report both:
- **Absolute rankings:** By theme_score (quality)
- **Volume rankings:** By publications_scored (quantity)

### Q: Can I have different numbers of topics per theme?

**A:** Yes! Themes don't need equal topic counts. Just adjust `topic_number` accordingly:
- Theme 1: 5 topics (topic_number 1-5)
- Theme 2: 3 topics (topic_number 1-3)
- Theme 3: 7 topics (topic_number 1-7)

### Q: What if topics have very different data completeness rates?

**A:** The hierarchical analysis script reports this:
- `data_completeness_rate` shown for each topic
- Parent theme summary shows average completeness
- Topics with low completeness (<70%) are flagged in the report

Consider running targeted data enrichment for low-completeness topics.

### Q: How do I compare topics across different parent themes?

**A:** Use the flat `topic_rankings_by_theme.csv` sorted by `theme_score` to see all topics ranked globally. Or create a cross-theme comparison:

```python
import pandas as pd
df = pd.read_csv('results/topic_rankings_by_theme.csv')
top_topics = df.nlargest(10, 'theme_score')
print(top_topics[['parent_theme', 'theme_name', 'theme_score', 'publications_scored']])
```

---

## Summary

✅ **15 independent searches** (3 themes × 5 topics) run by `run_theme_search.py`
✅ **Hierarchical grouping** by `analyze_hierarchical_themes.py`
✅ **Rankings within themes** showing topic priorities
✅ **Cross-theme comparison** showing overall parent theme strength
✅ **Flexible structure** supporting different topic counts and overlaps

This approach gives you fine-grained topic analysis while maintaining strategic theme-level view.
