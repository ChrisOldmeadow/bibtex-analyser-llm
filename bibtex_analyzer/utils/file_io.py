"""File I/O utilities for Bibtex Analyzer."""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import bibtexparser
from bibtexparser.bibdatabase import BibDatabase


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)


def load_bibtex(bib_path: str) -> List[Dict[str, Any]]:
    """Load and parse a BibTeX file.
    
    Args:
        bib_path: Path to the BibTeX file
        
    Returns:
        List of entry dictionaries
    """
    with open(bib_path, 'r', encoding='utf-8') as bibfile:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        db = bibtexparser.load(bibfile, parser=parser)
    
    entries = []
    for entry in db.entries:
        title = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        if abstract:
            entries.append({
                "id": entry.get("ID", ""),
                "title": title,
                "abstract": abstract,
                "year": entry.get("year", ""),
                "author": entry.get("author", ""),
                "journal": entry.get("journal", ""),
            })
    return entries


def save_tagged_entries(entries: List[Dict[str, Any]], out_path: str) -> None:
    """Save tagged entries to a CSV file.
    
    Args:
        entries: List of entry dictionaries
        out_path: Path to save the CSV file
    """
    df = pd.DataFrame(entries)
    df.to_csv(out_path, index=False)


def load_tagged_entries(csv_path: str) -> pd.DataFrame:
    """Load tagged entries from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing the tagged entries
    """
    return pd.read_csv(csv_path)


def save_wordcloud_data(tag_counts: Dict[str, int], output_dir: str) -> str:
    """Save word cloud data to a JSON file.
    
    Args:
        tag_counts: Dictionary mapping tags to counts
        output_dir: Directory to save the output file
        
    Returns:
        Path to the saved file
    """
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, "word_data.js")
    
    # Convert to list of [tag, count] pairs
    words_data = [[tag, count] for tag, count in tag_counts.items()]
    
    # Write as JavaScript variable
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("const wordList = ")
        json.dump(words_data, f, indent=2)
        f.write(";")
    
    return output_path


def save_html(content: str, output_path: str) -> str:
    """Save HTML content to a file.
    
    Args:
        content: HTML content to save
        output_path: Path to save the HTML file
        
    Returns:
        Path to the saved file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path
