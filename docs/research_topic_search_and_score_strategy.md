# Research Topic Search & Score Strategy (REVISED)

## Problem Statement
The current dashboard supports exact/fuzzy keyword matching plus embedding-based semantic/hybrid search. Exact/fuzzy methods work for short, distinct phrases but struggle with strategic multi-component topics (e.g., "Protect health with better access to affordable housing, healthy food, and clean water for vulnerable communities"). We need a repeatable plan to:
1. Use embedding/hybrid search to find papers for each complex topic without manual intervention.
2. Assign quality/relevance scores so papers and researchers can be ranked per topic.
3. Compare outputs across ~5 strategic topics efficiently and reproducibly.

## Constraints & Capabilities
- Bibliography already ingested (CSV/BibTeX with staff IDs, abstracts, etc.).
- Dashboard features: semantic search, hybrid rerank, staff summaries, CSV exports, optional LLM staff blurbs.
- **Existing scoring:** `HybridSemanticSearcher` already provides:
  - `embedding_score`: Semantic similarity (0-1) from text-embedding-3-small
  - `llm_relevance_score`: GPT-4o-mini relevance rating (0-10)
  - `llm_confidence`: GPT confidence in rating (0-10)
  - `llm_reasoning`: 1-2 sentence explanation
  - `llm_key_concepts`: List of matched concepts
  - `hybrid_score`: Weighted combination of embedding + LLM scores
- **Caching:** Both embedding and LLM caches minimize API costs for repeated searches.

## Recommended Solution: Automated Theme Pipeline

### Architecture Overview
Instead of manual dashboard clicks per topic, implement a batch theme runner that:
1. Reads theme definitions from `themes.yaml`
2. Executes hybrid search for each theme (leveraging existing LLM scoring)
3. **Calculates institutional-style theme scores** (SCImago methodology - see `theme_scoring_methodology.md`)
4. Generates per-theme paper lists, staff summaries, and cross-theme comparisons
5. Produces complete audit trail for reproducibility

### 1. Define Themes in YAML Configuration

**File:** `themes.yaml`

```yaml
themes:
  - id: housing_health_equity
    name: "Healthy Housing & Community Resilience"
    narrative: |
      Research examining how access to affordable housing, healthy food,
      clean water, and safe environments protects health outcomes for
      vulnerable communities including low-income families, elderly, and
      disaster-affected populations.
    # Search parameters
    semantic_threshold: 0.5      # Minimum embedding similarity (high recall)
    max_candidates: 100          # Top N papers to send for LLM analysis
    max_results: 20              # Final result set size
    min_llm_relevance: 6.0       # Minimum LLM score (0-10) to include

  - id: dementia_intervention
    name: "Dementia Awareness & Intervention"
    narrative: |
      Studies focused on Alzheimer's disease, cognitive decline, brain
      degeneration, including screening programs, early intervention
      strategies, caregiver support, and public awareness campaigns.
    semantic_threshold: 0.55
    max_candidates: 80
    max_results: 20
    min_llm_relevance: 6.5

  # Add more themes as needed...
```

**Key Design Decisions:**

- **No anchor phrases needed:** Modern embeddings capture semantic meaning from the full narrative; disconnected keyword lists may hurt coherence.
- **No separate rubric field:** The existing `llm_relevance_score` (0-10) and `llm_reasoning` already provide structured quality assessment. If custom rubrics are needed, they can be incorporated into the narrative.
- **Narrative only:** Pass the full strategic statement directly to hybrid search. The embedding model handles multi-component topics naturally.

### 2. Implementation: Theme Search Pipeline

**File:** `bibtex_analyzer/theme_search.py`

```python
"""Automated theme-based search pipeline for strategic topic analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import pandas as pd
from .semantic_search import HybridSemanticSearcher

logger = logging.getLogger(__name__)


class ThemeSearchPipeline:
    """Execute batch searches across multiple strategic themes."""

    def __init__(self, hybrid_searcher: HybridSemanticSearcher, df: pd.DataFrame):
        self.searcher = hybrid_searcher
        self.df = df

    def run_theme(self, theme: Dict, output_dir: Path) -> pd.DataFrame:
        """Execute hybrid search for a single theme.

        Args:
            theme: Theme definition from themes.yaml
            output_dir: Directory to save results

        Returns:
            DataFrame with search results and theme metadata
        """
        logger.info(f"Processing theme: {theme['name']}")
        logger.info(f"Query narrative: {theme['narrative'][:100]}...")

        # Use existing hybrid search - it already does everything we need!
        # Phase 1: Fast embedding filter across ALL papers
        # Phase 2: GPT-4o-mini analysis of top candidates (with caching)
        # Phase 3: Combined scoring with reasoning
        results_tuples = self.searcher.hybrid_search(
            query=theme['narrative'],  # Full narrative, no fragmentation
            df=self.df,
            threshold=theme['semantic_threshold'],
            max_embedding_candidates=theme['max_candidates'],
            max_results=theme['max_results'],
            logger=logger
        )

        if not results_tuples:
            logger.warning(f"No results found for theme: {theme['name']}")
            return pd.DataFrame()

        # Convert to DataFrame with enriched metadata
        result_rows = []
        analyzed_papers = self.searcher._last_analyzed_papers  # LLM analysis is cached here

        for analyzed_paper in analyzed_papers:
            idx = analyzed_paper['original_index']

            # Filter by minimum LLM relevance threshold
            llm_score = analyzed_paper.get('llm_relevance_score', 0)
            if llm_score < theme['min_llm_relevance']:
                continue

            # Add theme context to each paper
            paper = self.df.iloc[idx].to_dict()
            paper.update({
                'theme_id': theme['id'],
                'theme_name': theme['name'],
                'embedding_score': analyzed_paper['embedding_score'],
                'llm_relevance_score': llm_score,
                'llm_confidence': analyzed_paper.get('llm_confidence', 0),
                'llm_reasoning': analyzed_paper.get('llm_reasoning', ''),
                'llm_key_concepts': ', '.join(analyzed_paper.get('llm_key_concepts', [])),
                'hybrid_score': analyzed_paper.get('hybrid_score', 0)
            })
            result_rows.append(paper)

        results_df = pd.DataFrame(result_rows)

        # Save theme-specific papers
        theme_papers_path = output_dir / f"{theme['id']}_papers.csv"
        results_df.to_csv(theme_papers_path, index=False)
        logger.info(f"Saved {len(results_df)} papers to {theme_papers_path}")

        return results_df

    def aggregate_staff_for_theme(self, theme_df: pd.DataFrame, theme: Dict,
                                   output_dir: Path) -> pd.DataFrame:
        """Aggregate papers by staff member for a single theme.

        Args:
            theme_df: Papers for this theme
            theme: Theme definition
            output_dir: Output directory

        Returns:
            DataFrame with staff-level aggregates
        """
        if theme_df.empty or 'all_staff_ids' not in theme_df.columns:
            logger.warning(f"No staff data available for theme: {theme['name']}")
            return pd.DataFrame()

        # Explode staff IDs to get one row per staff-paper pair
        staff_papers = theme_df.copy()
        staff_papers['all_staff_ids'] = staff_papers['all_staff_ids'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        staff_papers = staff_papers.explode('all_staff_ids')
        staff_papers = staff_papers[staff_papers['all_staff_ids'].notna()]

        # Aggregate metrics per staff member
        staff_summary = staff_papers.groupby('all_staff_ids').agg({
            'ID': 'count',  # Total papers
            'llm_relevance_score': ['mean', 'max', 'std'],
            'hybrid_score': ['mean', 'max']
        }).reset_index()

        staff_summary.columns = [
            'staff_id',
            'paper_count',
            'avg_relevance', 'max_relevance', 'std_relevance',
            'avg_hybrid_score', 'max_hybrid_score'
        ]

        # Count high-quality papers (LLM score >= 8)
        high_quality = staff_papers[staff_papers['llm_relevance_score'] >= 8.0]
        high_quality_counts = high_quality.groupby('all_staff_ids').size().reset_index(name='high_quality_papers')
        staff_summary = staff_summary.merge(high_quality_counts, on='staff_id', how='left')
        staff_summary['high_quality_papers'] = staff_summary['high_quality_papers'].fillna(0).astype(int)

        # Sort by relevance and paper count
        staff_summary = staff_summary.sort_values(['avg_relevance', 'paper_count'], ascending=False)

        # Add theme metadata
        staff_summary['theme_id'] = theme['id']
        staff_summary['theme_name'] = theme['name']

        # Save staff summary
        staff_summary_path = output_dir / f"{theme['id']}_staff.csv"
        staff_summary.to_csv(staff_summary_path, index=False)
        logger.info(f"Saved staff summary for {len(staff_summary)} researchers to {staff_summary_path}")

        return staff_summary

    def run_all_themes(self, themes_path: Path, output_dir: Path,
                       generate_llm_summaries: bool = False) -> Dict:
        """Run all themes and generate comparative analysis.

        Args:
            themes_path: Path to themes.yaml
            output_dir: Directory for outputs
            generate_llm_summaries: Whether to generate narrative staff summaries

        Returns:
            Dictionary with theme statistics and paths to output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load themes configuration
        with open(themes_path, 'r') as f:
            config = yaml.safe_load(f)
        themes = config['themes']

        logger.info(f"Loaded {len(themes)} themes from {themes_path}")

        all_papers = []
        all_staff_summaries = []
        theme_stats = []

        for theme in themes:
            # Run search for this theme
            theme_df = self.run_theme(theme, output_dir)

            if theme_df.empty:
                continue

            # Generate staff summary
            staff_df = self.aggregate_staff_for_theme(theme_df, theme, output_dir)

            # Collect cross-theme statistics
            theme_stats.append({
                'theme_id': theme['id'],
                'theme_name': theme['name'],
                'publications': len(theme_df),  # Total publications matching theme
                'paper_count': len(theme_df),   # Alias for compatibility
                'unique_staff': len(staff_df),
                'avg_relevance': theme_df['llm_relevance_score'].mean(),
                'max_relevance': theme_df['llm_relevance_score'].max(),
                'high_quality_papers': len(theme_df[theme_df['llm_relevance_score'] >= 8.0]),
                'top_5_staff': ', '.join(staff_df.head(5)['staff_id'].astype(str).tolist())
            })

            all_papers.append(theme_df)
            if not staff_df.empty:
                all_staff_summaries.append(staff_df)

        # Create cross-theme comparison
        comparison_df = pd.DataFrame(theme_stats)
        comparison_path = output_dir / "theme_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved theme comparison to {comparison_path}")

        # Optionally combine all results
        if all_papers:
            combined_papers = pd.concat(all_papers, ignore_index=True)
            combined_papers_path = output_dir / "all_themes_papers.csv"
            combined_papers.to_csv(combined_papers_path, index=False)
            logger.info(f"Saved combined papers ({len(combined_papers)} total) to {combined_papers_path}")

        if all_staff_summaries:
            combined_staff = pd.concat(all_staff_summaries, ignore_index=True)
            combined_staff_path = output_dir / "all_themes_staff.csv"
            combined_staff.to_csv(combined_staff_path, index=False)
            logger.info(f"Saved combined staff summaries to {combined_staff_path}")

        return {
            'themes': theme_stats,
            'output_dir': str(output_dir),
            'comparison_file': str(comparison_path)
        }


def load_themes(themes_path: Path) -> List[Dict]:
    """Load and validate theme definitions."""
    with open(themes_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['themes']
```

### 3. CLI Interface

**Usage:**
```bash
# Run all themes from configuration file
python -m bibtex_analyzer theme-search \
  --themes themes.yaml \
  --dataset results.csv \
  --output output/theme_results/

# Or with BibTeX input
python -m bibtex_analyzer theme-search \
  --themes themes.yaml \
  --input references.bib \
  --output output/theme_results/
```

**Output Structure:**
```
output/theme_results/
├── theme_comparison.csv          # Cross-theme stats with SCImago-style scores
├── theme_scores_detail.csv       # Detailed score components per theme
├── all_themes_papers.csv         # All papers from all themes
├── all_themes_staff.csv          # All staff summaries
├── housing_health_equity_papers.csv
├── housing_health_equity_staff.csv
├── dementia_intervention_papers.csv
└── dementia_intervention_staff.csv
```

**Theme Comparison CSV includes:**
- `publications`: Total number of publications matching the theme
- `theme_score`: Overall score (0-100) using SCImago methodology
- `research_score`: Research performance component (0-100)
- `societal_score`: Societal impact component (0-100)
- `unique_staff`: Number of distinct staff members contributing to theme
- `avg_relevance`: Average LLM relevance score (0-10)
- `high_quality_papers`: Count of papers with relevance score ≥ 8.0
- Detailed metrics: Q1%, international collaboration%, normalized impact, excellence rate, etc.

See **`theme_scoring_methodology.md`** for complete scoring algorithm details.

### 4. Dashboard Integration (Optional)

Add a "Theme Search" tab to the dashboard with:
- **Upload themes.yaml** file upload
- **Select dataset** from dropdown (existing datasets)
- **Run Batch Search** button triggers pipeline
- **Progress indicator** showing current theme being processed
- **Results table** showing theme comparison stats
- **Download buttons** for each theme's results

## Advantages Over Manual Workflow

| Aspect | Manual Dashboard Clicks | Automated Pipeline |
|--------|------------------------|-------------------|
| **LLM Scoring** | Single pass (built-in) | Single pass (built-in) |
| **Query Format** | Narrative only | Narrative only |
| **API Cost** | Optimal (uses caching) | Optimal (uses caching) |
| **Reproducibility** | Manual, error-prone | Config file defines everything |
| **Audit Trail** | None | Full YAML + logs |
| **Effort per Run** | 5× manual searches | One command |
| **Parameter Consistency** | Risk of variation | Guaranteed identical |
| **Theme Comparison** | Manual CSV merging | Automatic aggregation |
| **Time to Results** | ~30-60 min manual work | ~5-10 min automated |

## Key Technical Insights

### Why Anchor Phrases Are Not Needed
- Modern embedding models (text-embedding-3-small) capture **semantic meaning**, not keyword matching
- The narrative "affordable housing + healthy food + vulnerable communities" already embeds as a coherent concept
- Adding disconnected keyword lists may **reduce semantic coherence**
- The LLM reranking step evaluates conceptual overlap explicitly

### Why Double-Scoring Is Wasteful
The existing `HybridSemanticSearcher.hybrid_search()` already provides:
1. **Fast embedding filter** (Phase 1): Finds semantically similar papers
2. **LLM relevance analysis** (Phase 2): GPT-4o-mini scores each paper with reasoning
3. **Combined scoring** (Phase 3): Weighted hybrid score

Running a second LLM scoring pass would:
- Double API costs unnecessarily
- Introduce potential inconsistencies between two LLM calls
- Not improve quality (same model, similar prompts)

### Customizing LLM Scoring Rubrics (If Needed)
If you need custom 0-5 rubrics per theme instead of the default 0-10 relevance score:

1. **Add rubric to theme definition:**
   ```yaml
   - id: housing_health_equity
     rubric: |
       5 = Direct study of housing/food/water interventions for vulnerable groups
       4 = Strong focus on one intervention + health outcomes
       3 = Related social determinants, indirect evidence
       2 = Tangential mention of topics
       1 = Minimal relevance
   ```

2. **Modify LLM prompt in `HybridSemanticSearcher.llm_analyze_relevance()`:**
   ```python
   prompt = f"""Analyze this research paper's relevance to the theme.

   Theme: "{query}"

   Rubric:
   {rubric_text}

   Paper:
   Title: {title}
   Abstract: {abstract}

   Provide analysis in JSON format:
   {{"score": 0-5, "rationale": "...", "key_concepts": [...]}}
   """
   ```

## Notes for Implementation

- **Caching is critical:** Both `.embeddings_cache/` and `.llm_relevance_cache/` save significant costs on repeated runs
- **Cache invalidation:** Caches are keyed by model + text content, so changing models requires new cache
- **Progress logging:** Use the existing logger infrastructure for real-time feedback
- **Error handling:** Failed LLM calls default to neutral scores (5/10) rather than crashing
- **Staff ID handling:** Use `all_staff_ids` field (array) to properly credit co-authors in multi-staff papers
