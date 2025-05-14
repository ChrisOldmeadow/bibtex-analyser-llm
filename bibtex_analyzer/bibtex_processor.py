import bibtexparser
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

class BibtexProcessor:
    def __init__(self):
        self.entries = []

    def load_entries(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entries from a BibTeX file."""
        with open(file_path) as bibtex_file:
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
                filtered = [e for e in filtered if int(e.get('year', 0)) >= value]
            elif key == 'max_year':
                filtered = [e for e in filtered if int(e.get('year', 9999)) <= value]
            else:
                filtered = [e for e in filtered if e.get(key) == value]
        return filtered

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
    Process a BibTeX file and optionally save to CSV.
    
    Args:
        input_file: Path to the input BibTeX file
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
