import bibtexparser
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import os

class BibtexProcessor:
    def __init__(self):
        self.entries = []
    
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
    
    def _load_csv_entries(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entries from a CSV file with bibliography data."""
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to list of dictionaries
            entries = df.to_dict('records')
            
            # Normalize entries to match expected format
            normalized_entries = []
            for entry in entries:
                # Remove NaN values and convert to strings
                clean_entry = {}
                for key, value in entry.items():
                    if pd.notna(value):
                        clean_entry[key.lower()] = str(value)
                    else:
                        clean_entry[key.lower()] = ''
                
                # Ensure required fields exist
                if 'id' not in clean_entry and 'ID' not in clean_entry:
                    # Generate an ID if none exists
                    clean_entry['ID'] = f"entry_{len(normalized_entries) + 1}"
                elif 'id' in clean_entry:
                    clean_entry['ID'] = clean_entry['id']
                
                normalized_entries.append(clean_entry)
            
            return normalized_entries
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

    def load_entries(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entries from a BibTeX or CSV file."""
        file_format = self._detect_file_format(file_path)
        
        if file_format == 'csv':
            self.entries = self._load_csv_entries(file_path)
        else:  # bibtex
            with open(file_path, 'r', encoding='utf-8') as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)
                self.entries = bib_database.entries
        
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
    
    def _extract_year(self, year_value: str) -> int:
        """Extract year as integer from various year formats."""
        if not year_value:
            return 0
        
        # Convert to string and clean up
        year_str = str(year_value).strip()
        
        # Handle various formats
        import re
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
