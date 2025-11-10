#!/usr/bin/env python3
"""
Enrich publication data with OpenAlex metrics via DOI lookup.

Fetches for each publication:
- Field-Weighted Citation Impact (cited_by_percentile_year as proxy for FWCI)
- Citation count (for validation/backup)
- Open Access status
- Publication type
- Topics/concepts with relevance scores
- Authorships with position info

This replaces manual baseline calculation with global benchmarks from OpenAlex.

Usage:
    python scripts/enrich_with_openalex.py \\
        --input "HMRI Pub Abstracts_20250703.csv" \\
        --output data/hmri_enriched.csv \\
        --cache data/openalex_cache.json
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import requests
from collections import defaultdict

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bibtex_analyzer.bibtex_processor import BibtexProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAlex API configuration
OPENALEX_API_BASE = "https://api.openalex.org/works"
EMAIL = "your-email@example.com"  # Replace with your email for polite API usage
HEADERS = {
    "User-Agent": f"HMRI Theme Analysis (mailto:{EMAIL})",
    "Accept": "application/json"
}
RATE_LIMIT_DELAY = 0.1  # 100ms between requests (polite rate limiting)


class OpenAlexEnricher:
    """Enrich publication data with OpenAlex metrics."""

    def __init__(self, cache_path: Optional[Path] = None):
        """Initialize enricher with optional cache.

        Args:
            cache_path: Path to JSON cache file for OpenAlex responses
        """
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.stats = {
            'total': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'not_found': 0,
            'errors': 0,
            'enriched': 0
        }

    def _load_cache(self) -> Dict:
        """Load OpenAlex response cache from disk."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries from {self.cache_path}")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save OpenAlex response cache to disk."""
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                logger.info(f"Saved cache with {len(self.cache)} entries to {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def fetch_openalex_data(self, doi: str) -> Optional[Dict]:
        """Fetch publication data from OpenAlex by DOI.

        Args:
            doi: DOI of the publication

        Returns:
            OpenAlex work data or None if not found
        """
        # Check cache first
        if doi in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[doi]

        # Fetch from API
        self.stats['api_calls'] += 1

        # Clean DOI
        clean_doi = doi.strip().lower()
        if not clean_doi.startswith('http'):
            clean_doi = f"https://doi.org/{clean_doi}"

        try:
            url = f"{OPENALEX_API_BASE}/{clean_doi}"
            response = requests.get(url, headers=HEADERS, timeout=10)

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

            if response.status_code == 200:
                data = response.json()
                self.cache[doi] = data
                return data
            elif response.status_code == 404:
                self.stats['not_found'] += 1
                self.cache[doi] = None  # Cache 404s to avoid re-querying
                return None
            else:
                logger.warning(f"OpenAlex API error {response.status_code} for DOI: {doi}")
                self.stats['errors'] += 1
                return None

        except Exception as e:
            logger.warning(f"Error fetching DOI {doi}: {e}")
            self.stats['errors'] += 1
            return None

    def extract_metrics(self, openalex_data: Dict) -> Dict:
        """Extract relevant metrics from OpenAlex response.

        Args:
            openalex_data: OpenAlex work JSON

        Returns:
            Dictionary with extracted metrics
        """
        metrics = {
            'openalex_id': openalex_data.get('id', ''),
            'openalex_cited_by_count': openalex_data.get('cited_by_count', 0),
        }

        # Field-Weighted Citation Impact proxy
        # OpenAlex provides cited_by_percentile_year which shows where the paper
        # ranks in citations compared to others published the same year
        # Values: {"min": 0, "max": 100} where 50 = median
        cited_by_percentile = openalex_data.get('cited_by_percentile_year', {})
        if cited_by_percentile and 'max' in cited_by_percentile:
            # Convert percentile to FWCI-like score
            # Percentile 50 = median = FWCI 1.0
            # Percentile 90 ≈ FWCI 2.0-3.0
            # This is an approximation
            percentile = cited_by_percentile.get('max', 50)
            metrics['openalex_citation_percentile'] = percentile

            # Rough FWCI approximation (will be better than raw citations)
            if percentile >= 95:
                fwci_approx = 3.0
            elif percentile >= 90:
                fwci_approx = 2.5
            elif percentile >= 75:
                fwci_approx = 1.5
            elif percentile >= 50:
                fwci_approx = 1.0
            elif percentile >= 25:
                fwci_approx = 0.7
            else:
                fwci_approx = 0.5
            metrics['openalex_fwci_approx'] = fwci_approx
        else:
            metrics['openalex_citation_percentile'] = None
            metrics['openalex_fwci_approx'] = 1.0  # Neutral

        # Open Access status
        oa_info = openalex_data.get('open_access', {})
        metrics['openalex_is_oa'] = oa_info.get('is_oa', False)
        metrics['openalex_oa_status'] = oa_info.get('oa_status', '')  # gold, green, bronze, closed

        # Publication type
        metrics['openalex_type'] = openalex_data.get('type', '')  # article, review, etc.

        # Concepts/Topics (top 3)
        concepts = openalex_data.get('concepts', [])
        top_concepts = sorted(concepts, key=lambda x: x.get('score', 0), reverse=True)[:3]
        metrics['openalex_concepts'] = ', '.join([c.get('display_name', '') for c in top_concepts])

        # Authorships - check for institutional presence
        authorships = openalex_data.get('authorships', [])
        metrics['openalex_author_count'] = len(authorships)

        # Check for corresponding author
        has_corresponding = any(
            auth.get('author', {}).get('orcid') and auth.get('is_corresponding')
            for auth in authorships
        )
        metrics['openalex_has_corresponding'] = has_corresponding

        # International collaboration (multiple countries)
        countries = set()
        for auth in authorships:
            for inst in auth.get('institutions', []):
                country = inst.get('country_code')
                if country:
                    countries.add(country)
        metrics['openalex_countries'] = ', '.join(sorted(countries))
        metrics['openalex_is_international'] = len(countries) > 1

        return metrics

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich DataFrame with OpenAlex data.

        Args:
            df: DataFrame with publications (must have 'doi' or 'Ref_DOI' column)

        Returns:
            Enriched DataFrame with OpenAlex metrics
        """
        self.stats['total'] = len(df)
        logger.info(f"Enriching {len(df)} publications with OpenAlex data")

        # Find DOI column
        doi_col = None
        for col in ['doi', 'Ref_DOI', 'DOI']:
            if col in df.columns:
                doi_col = col
                break

        if not doi_col:
            raise ValueError("No DOI column found in DataFrame")

        logger.info(f"Using DOI column: {doi_col}")

        # Initialize new columns
        openalex_cols = [
            'openalex_id', 'openalex_cited_by_count', 'openalex_citation_percentile',
            'openalex_fwci_approx', 'openalex_is_oa', 'openalex_oa_status',
            'openalex_type', 'openalex_concepts', 'openalex_author_count',
            'openalex_has_corresponding', 'openalex_countries', 'openalex_is_international'
        ]

        enriched_rows = []

        for idx, row in df.iterrows():
            doi = row.get(doi_col)

            if pd.notna(doi) and str(doi).strip():
                # Fetch OpenAlex data
                openalex_data = self.fetch_openalex_data(str(doi).strip())

                if openalex_data:
                    metrics = self.extract_metrics(openalex_data)
                    self.stats['enriched'] += 1
                else:
                    # No data found - use defaults
                    metrics = {col: None for col in openalex_cols}
            else:
                # No DOI - use defaults
                metrics = {col: None for col in openalex_cols}

            # Merge with original row
            enriched_row = row.to_dict()
            enriched_row.update(metrics)
            enriched_rows.append(enriched_row)

            # Progress logging
            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx + 1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%) - "
                          f"Cache: {self.stats['cache_hits']}, API: {self.stats['api_calls']}, "
                          f"Enriched: {self.stats['enriched']}")

        enriched_df = pd.DataFrame(enriched_rows)

        # Save cache
        self._save_cache()

        return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description="Enrich publication data with OpenAlex metrics"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to institutional publications CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to save enriched CSV'
    )
    parser.add_argument(
        '--cache',
        type=Path,
        default=Path('data/openalex_cache.json'),
        help='Path to OpenAlex cache file (default: data/openalex_cache.json)'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Your email for OpenAlex API (polite usage)'
    )

    args = parser.parse_args()

    # Update email if provided
    if args.email:
        global EMAIL, HEADERS
        EMAIL = args.email
        HEADERS["User-Agent"] = f"HMRI Theme Analysis (mailto:{EMAIL})"

    logger.info("\n" + "="*60)
    logger.info("OPENALEX DATA ENRICHMENT")
    logger.info("="*60 + "\n")

    try:
        # Load and deduplicate data
        logger.info("Loading publication data...")
        processor = BibtexProcessor()
        entries = processor.load_entries(str(args.input), deduplicate=True)
        df = pd.DataFrame(entries)

        logger.info(f"Loaded {len(df)} unique publications")
        logger.info(f"Original entries: {processor.deduplication_stats['total_entries']}")
        logger.info(f"Duplicates removed: {processor.deduplication_stats['duplicates_removed']}")

        # Check for DOI column
        doi_cols = [col for col in df.columns if 'doi' in col.lower()]
        logger.info(f"DOI columns found: {doi_cols}")

        if not doi_cols:
            logger.error("No DOI column found! Cannot enrich without DOIs.")
            sys.exit(1)

        # Count papers with DOIs
        doi_col = doi_cols[0]
        has_doi = df[doi_col].notna().sum()
        logger.info(f"Papers with DOI: {has_doi}/{len(df)} ({has_doi/len(df)*100:.1f}%)")

        if has_doi == 0:
            logger.error("No papers have DOIs! Cannot proceed with enrichment.")
            sys.exit(1)

        # Enrich with OpenAlex
        logger.info("\nStarting OpenAlex enrichment...")
        logger.info(f"Cache file: {args.cache}")
        logger.info(f"Rate limit: {1/RATE_LIMIT_DELAY:.0f} requests/second")
        logger.info(f"Estimated time: ~{has_doi * RATE_LIMIT_DELAY / 60:.1f} minutes for new lookups\n")

        enricher = OpenAlexEnricher(cache_path=args.cache)
        enriched_df = enricher.enrich_dataframe(df)

        # Save enriched data
        args.output.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.to_csv(args.output, index=False)

        # Report statistics
        logger.info("\n" + "="*60)
        logger.info("ENRICHMENT COMPLETE")
        logger.info("="*60)
        logger.info(f"Total publications: {enricher.stats['total']}")
        logger.info(f"Successfully enriched: {enricher.stats['enriched']}")
        logger.info(f"Cache hits: {enricher.stats['cache_hits']}")
        logger.info(f"API calls: {enricher.stats['api_calls']}")
        logger.info(f"Not found: {enricher.stats['not_found']}")
        logger.info(f"Errors: {enricher.stats['errors']}")
        logger.info(f"\nEnriched data saved to: {args.output}")
        logger.info(f"Cache saved to: {args.cache}")
        logger.info("="*60 + "\n")

        # Show sample enrichment
        sample = enriched_df[enriched_df['openalex_id'].notna()].head(3)
        if not sample.empty:
            logger.info("Sample enriched records:")
            for idx, row in sample.iterrows():
                logger.info(f"\n  Title: {row.get('title', 'N/A')[:60]}...")
                logger.info(f"  FWCI approx: {row.get('openalex_fwci_approx', 'N/A')}")
                logger.info(f"  Citation percentile: {row.get('openalex_citation_percentile', 'N/A')}")
                logger.info(f"  Open Access: {row.get('openalex_is_oa', 'N/A')}")
                logger.info(f"  International: {row.get('openalex_is_international', 'N/A')}")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
