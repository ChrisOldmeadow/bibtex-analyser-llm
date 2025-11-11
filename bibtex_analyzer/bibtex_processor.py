import bibtexparser
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import os
import re

class BibtexProcessor:
    def __init__(self):
        self.entries = []
        self.deduplication_stats = self._init_deduplication_stats()
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect the file format based on extension and content."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return 'csv'
        elif file_ext == '.bib':
            return 'bibtex'
        else:
            # Try to detect based on content for files without clear extensions
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    if content.strip().startswith('@'):
                        return 'bibtex'
                    elif ',' in content and ('title' in content.lower() or 'author' in content.lower()):
                        return 'csv'
            except Exception:
                pass
            
            # Default to bibtex for backward compatibility
            return 'bibtex'
    
    def _init_deduplication_stats(self) -> Dict[str, Any]:
        """Initialize blank deduplication statistics."""
        return {
            'total_entries': 0,
            'unique_entries': 0,
            'duplicates_removed': 0,
            'staff_contributors': {}
        }

    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Return deduplication statistics from the last load."""
        return self.deduplication_stats
    
    def _read_csv_with_fallback(self, file_path: str) -> pd.DataFrame:
        """Read CSV trying UTF-8 first and falling back to common legacy encodings."""
        encodings = ['utf-8', 'latin-1', 'windows-1252']
        last_error = None
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f"Unable to decode CSV using {encodings}: {last_error.reason}"
        ) if last_error else ValueError("Unknown error reading CSV file")

    def _load_csv_entries(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entries from a CSV file with bibliography data."""
        try:
            df = self._read_csv_with_fallback(file_path)
            
            # Convert DataFrame to list of dictionaries
            entries = df.to_dict('records')
            
            # Normalize entries to match expected format
            normalized_entries = []
            for entry in entries:
                # Remove NaN values and convert to strings
                clean_entry: Dict[str, Any] = {}
                for key, value in entry.items():
                    normalized_key = str(key)
                    if pd.notna(value):
                        clean_entry[normalized_key] = value if isinstance(value, str) else str(value)
                    else:
                        clean_entry[normalized_key] = ''
                
                # Ensure required fields exist
                if 'ID' not in clean_entry and 'id' not in clean_entry:
                    # Generate an ID if none exists
                    clean_entry['ID'] = f"entry_{len(normalized_entries) + 1}"
                elif 'id' in clean_entry and not clean_entry.get('ID'):
                    clean_entry['ID'] = clean_entry['id']

                # Provide canonical title-case aliases for commonly used fields
                lower_keys = {k.lower(): k for k in clean_entry.keys()}

                def assign_alias(source_key: str, alias: str):
                    src_lower = source_key.lower()
                    existing_key = lower_keys.get(src_lower)
                    if existing_key and alias not in clean_entry:
                        clean_entry[alias] = clean_entry[existing_key]

                alias_fields = {
                    'publication_id': 'Publication_ID',
                    'reported_year': 'Reported_Year',
                    'output_title': 'Output_Title',
                    'output_abstract': 'Output_Abstract',
                    'nuro_abstract': 'NURO_abstract',
                    'numberplate': 'NumberPlate',
                    'staff_id': 'Staff_ID',
                    'all_staff_ids': 'all_staff_ids',
                    'abstract': 'abstract',
                    'title': 'title'
                }
                for source, alias in alias_fields.items():
                    assign_alias(source, alias)

                # Populate canonical lowercase fields for downstream consumers
                def _first_non_empty(keys: List[str]) -> Optional[str]:
                    for key in keys:
                        key_lower = key.lower()
                        actual_key = lower_keys.get(key_lower)
                        if actual_key and actual_key in clean_entry:
                            val = clean_entry[actual_key]
                            if val is not None:
                                text = str(val).strip()
                                if text and text.lower() != 'nan':
                                    return text
                    return None

                title_value = _first_non_empty(['title', 'Title', 'Output_Title', 'output_title'])
                if title_value:
                    clean_entry['title'] = title_value
                    clean_entry['Title'] = title_value
                elif 'title' not in clean_entry:
                    clean_entry['title'] = ''

                abstract_value = _first_non_empty(['abstract', 'Abstract', 'Output_Abstract',
                                                   'output_abstract', 'NURO_abstract', 'nuro_abstract'])
                if abstract_value:
                    clean_entry['abstract'] = abstract_value
                elif 'abstract' not in clean_entry:
                    clean_entry['abstract'] = ''

                year_value = _first_non_empty(['year', 'Year', 'Reported_Year', 'reported_year'])
                if year_value:
                    clean_entry['year'] = year_value
                    clean_entry['Reported_Year'] = year_value
                elif 'year' not in clean_entry:
                    clean_entry['year'] = ''

                journal_value = _first_non_empty([
                    'journal', 'article_journal', 'output_journal', 'parent-title', 'parent_title'
                ])
                if journal_value:
                    clean_entry['journal'] = journal_value
                elif 'journal' not in clean_entry:
                    clean_entry['journal'] = ''

                doi_value = _first_non_empty(['doi', 'DOI', 'ref_doi', 'Ref_DOI'])
                if doi_value:
                    clean_entry['doi'] = doi_value
                elif 'doi' not in clean_entry:
                    clean_entry['doi'] = ''
                
                normalized_entries.append(clean_entry)
            
            return normalized_entries
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

    def load_entries(self, file_path: str, deduplicate: bool = False,
                     dedup_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load entries from a BibTeX or CSV file with optional deduplication."""
        self.deduplication_stats = self._init_deduplication_stats()
        file_format = self._detect_file_format(file_path)
        
        if file_format == 'csv':
            self.entries = self._load_csv_entries(file_path)
        else:  # bibtex
            with open(file_path, 'r', encoding='utf-8') as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)
                self.entries = bib_database.entries

        if deduplicate:
            self.entries = self._deduplicate_entries(self.entries, dedup_fields=dedup_fields)
        else:
            self.deduplication_stats['total_entries'] = len(self.entries)
            self.deduplication_stats['unique_entries'] = len(self.entries)

        return self.entries

    def get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get an entry by its ID."""
        for entry in self.entries:
            if entry.get('ID') == entry_id:
                return entry
        return None

    def filter_entries(self, **filters) -> List[Dict[str, Any]]:
        """Filter entries based on given criteria."""
        filtered = self.entries
        for key, value in filters.items():
            if key == 'min_year':
                filtered = [e for e in filtered if self._extract_year(e.get('year', '')) >= value]
            elif key == 'max_year':
                filtered = [e for e in filtered if self._extract_year(e.get('year', '')) <= value]
            else:
                filtered = [e for e in filtered if e.get(key) == value]
        return filtered

    def _parse_staff_value(self, value: Any) -> List[str]:
        """Normalize staff fields into a list of IDs."""
        if value is None:
            return []
        if isinstance(value, list):
            iterable = value
        else:
            text = str(value).strip()
            if not text:
                return []
            if text.startswith('[') and text.endswith(']'):
                inner = text[1:-1]
                iterable = [v.strip() for v in inner.split(',') if v.strip()]
            else:
                iterable = [v.strip() for v in text.replace(';', ',').split(',')]
        cleaned = []
        for val in iterable:
            clean = str(val).strip().strip("'\"")
            if clean and clean.lower() != 'nan':
                cleaned.append(clean)
        return cleaned

    def _normalize_title(self, title: str) -> str:
        """Normalize a title string for deduplication."""
        if not title:
            return ''
        text = re.sub(r'\s+', ' ', str(title).lower())
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def _get_field(self, entry: Dict[str, Any], field_names: List[str]) -> Optional[str]:
        """Return the first non-empty value across case variants."""
        for name in field_names:
            candidates = [name, name.lower(), name.upper()]
            for candidate in candidates:
                if candidate in entry:
                    value = entry[candidate]
                    if value is not None and str(value).strip():
                        return str(value).strip()
        return None

    def _extract_staff_ids(self, entry: Dict[str, Any]) -> List[str]:
        """Extract staff IDs from known staff fields."""
        staff_fields = ['all_staff_ids', 'NumberPlate', 'numberplate', 'staff_id', 'Staff_ID']
        collected: List[str] = []
        for field in staff_fields:
            if field in entry:
                collected.extend(self._parse_staff_value(entry[field]))
        return sorted(set(collected))

    def _merge_entries(self, base: Dict[str, Any], new_entry: Dict[str, Any]) -> None:
        """Merge non-empty values and staff IDs from new_entry into base."""
        for key, value in new_entry.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            if key not in base or base[key] in ['', None] or (isinstance(base[key], float) and pd.isna(base[key])):
                base[key] = value

        existing_staff = self._parse_staff_value(base.get('all_staff_ids'))
        new_staff = self._extract_staff_ids(new_entry)
        combined = sorted(set(existing_staff).union(new_staff))
        if combined:
            base['all_staff_ids'] = combined

    def _build_dedup_key(self, entry: Dict[str, Any], fallback_idx: int,
                         dedup_fields: Optional[List[str]] = None) -> Tuple[str, str]:
        """Generate a deduplication key using priority fields."""
        if dedup_fields:
            for field in dedup_fields:
                value = self._get_field(entry, [field])
                if value:
                    return (f"custom::{field.lower()}::{value}", 'custom')

        publication_id = self._get_field(entry, ['Publication_ID', 'ID', 'publication_id', 'id'])
        if publication_id:
            return (f"id::{publication_id}", 'id')

        doi = self._get_field(entry, ['ref_doi', 'doi', 'Ref_DOI'])
        if doi:
            return (f"doi::{doi.lower()}", 'doi')

        title = self._get_field(entry, ['Output_Title', 'title', 'Title'])
        normalized_title = self._normalize_title(title) if title else ''
        if normalized_title:
            return (f"title::{normalized_title}", 'title')

        return (f"row::{fallback_idx}", 'row')

    def _deduplicate_entries(self, entries: List[Dict[str, Any]],
                              dedup_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Deduplicate entries and track staff contributors."""
        stats = self.deduplication_stats
        stats['total_entries'] = len(entries)

        canonical_entries: Dict[str, Dict[str, Any]] = {}
        staff_map: Dict[str, List[str]] = {}

        for idx, entry in enumerate(entries):
            key, _ = self._build_dedup_key(entry, idx, dedup_fields)
            if key not in canonical_entries:
                canonical_entries[key] = entry.copy()
            else:
                self._merge_entries(canonical_entries[key], entry)

            staff_ids = self._extract_staff_ids(entry)
            if staff_ids:
                existing = staff_map.setdefault(key, [])
                seen = set(existing)
                for staff_id in staff_ids:
                    if staff_id not in seen:
                        existing.append(staff_id)
                        seen.add(staff_id)

        deduplicated = list(canonical_entries.values())
        stats['unique_entries'] = len(deduplicated)
        stats['duplicates_removed'] = stats['total_entries'] - stats['unique_entries']
        stats['staff_contributors'] = staff_map

        # Ensure all canonical entries include the complete staff list
        for key, entry in canonical_entries.items():
            contributors = staff_map.get(key, [])
            entry['all_staff_ids'] = sorted(set(contributors))

        return deduplicated
    
    def _extract_year(self, year_value: str) -> int:
        """Extract year as integer from various year formats."""
        if not year_value:
            return 0
        
        # Convert to string and clean up
        year_str = str(year_value).strip()
        
        # Handle various formats
        # Extract 4-digit year from strings like "2023", "2023-01-01", "c2023", etc.
        # Only accept years that look reasonable (1000-9999)
        year_match = re.search(r'(\d{4})', year_str)
        if year_match:
            year = int(year_match.group(1))
            # Only accept reasonable years (not too old, not too far in future)
            if 1000 <= year <= 2100:
                return year
        
        # Fallback: try to convert directly, but only for reasonable years
        try:
            year = int(float(year_str))
            if 1000 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass
        
        return 0

    def export_to_csv(self, output_path: str) -> str:
        """Export entries to a CSV file."""
        df = pd.DataFrame(self.entries)
        df.to_csv(output_path, index=False)
        return output_path

    def get_field_values(self, field: str, unique: bool = True) -> List[Any]:
        """Get all values for a specific field."""
        values = [entry.get(field) for entry in self.entries if field in entry]
        return list(set(values)) if unique else values

def process_bibtex_file(input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process a BibTeX or CSV file and optionally save to CSV.
    
    Args:
        input_file: Path to the input BibTeX or CSV file
        output_file: Optional path to save the output CSV
    
    Returns:
        List of processed entries with normalized field names
    """
    # Initialize the processor and load entries
    processor = BibtexProcessor()
    entries = processor.load_entries(input_file)
    
    # Normalize field names and handle different abstract field names
    normalized_entries = []
    abstract_fields = ['abstract', 'annote', 'note']  # Common field names that might contain abstracts
    
    for entry in entries:
        # Create a new entry with lowercase keys for case-insensitive access
        normalized_entry = {k.lower(): v for k, v in entry.items()}
        
        # Try to find the abstract in common fields
        if 'abstract' not in normalized_entry or not normalized_entry['abstract']:
            for field in abstract_fields:
                if field in normalized_entry and normalized_entry[field]:
                    normalized_entry['abstract'] = normalized_entry[field]
                    break
        
        # Ensure all entries have an abstract field, even if empty
        if 'abstract' not in normalized_entry:
            normalized_entry['abstract'] = ''
            
        # Preserve the original case of the entry ID
        if 'ID' in entry:
            normalized_entry['ID'] = entry['ID']
        elif 'id' in normalized_entry:
            normalized_entry['ID'] = normalized_entry.pop('id')
            
        normalized_entries.append(normalized_entry)
    
    if output_file:
        # Create a DataFrame with normalized field names for export
        df = pd.DataFrame(normalized_entries)
        df.to_csv(output_file, index=False)
    
    return normalized_entries
