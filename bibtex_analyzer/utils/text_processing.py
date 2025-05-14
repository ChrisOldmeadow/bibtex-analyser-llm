"""Text processing utilities for Bibtex Analyzer."""

from typing import List, Set, Dict, Any
import re


def normalize_tags(tag_list: List[str], style: str = "title") -> List[str]:
    """Normalize a list of tags.
    
    Args:
        tag_list: List of tags to normalize
        style: Normalization style ('title', 'lower', or 'upper')
        
    Returns:
        List of normalized tags
    """
    cleaned = set()
    for tag in tag_list:
        tag = tag.strip()
        if not tag:
            continue
            
        if style == "lower":
            tag = tag.lower()
        elif style == "title":
            tag = tag.title()
        elif style == "upper":
            tag = tag.upper()
            
        cleaned.add(tag)
    return sorted(cleaned)


def slugify(text: str) -> str:
    """Convert a string to a URL-friendly slug.
    
    Args:
        text: Text to convert
        
    Returns:
        URL-friendly slug
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().replace(" ", "-")
    # Remove all non-word characters (keeps only letters, numbers, and hyphens)
    slug = re.sub(r'[^\w\-]', '', slug)
    # Replace multiple hyphens with a single hyphen
    slug = re.sub(r'\-+', '-', slug)
    return slug


def clean_text(text: str) -> str:
    """Clean and normalize text for processing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters, keeping letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s.,;:!?\-]', ' ', text)
    return text.strip()


def collapse_tags(tag_list: List[str], synonym_map: Dict[str, str] = None) -> List[str]:
    """Collapse similar or redundant tags using a synonym map.
    
    Args:
        tag_list: List of tags to process
        synonym_map: Dictionary mapping variants to canonical forms
        
    Returns:
        List of collapsed tags
    """
    if synonym_map is None:
        synonym_map = {
            "Rct": "Randomised Controlled Trial",
            "RCT": "Randomised Controlled Trial",
            "Intervention Study": "Randomised Controlled Trial",
            "Pragmatic Trial": "Randomised Controlled Trial",
            "Mixed-Effects Models": "Mixed Models",
            "Children": "School Children",
            "Smoking": "Smokers"
        }
    
    collapsed = []
    for tag in tag_list:
        tag = tag.strip()
        if not tag:
            continue
        # Use the canonical form if it exists, otherwise use the original tag
        canonical_tag = synonym_map.get(tag, tag)
        collapsed.append(canonical_tag)
    
    return list(set(collapsed))  # Remove duplicates


def filter_redundant_tags(tag_counts: Dict[str, int], drop_if_subset_of: set = None) -> Dict[str, int]:
    """Filter out redundant tags based on string containment.
    
    For example, if 'Randomised Controlled Trial' and 'Pragmatic Trial' are both present,
    and 'Pragmatic Trial' is in the drop_if_subset_of set, it will be removed.
    
    Args:
        tag_counts: Dictionary mapping tags to their counts
        drop_if_subset_of: Set of tags to check for containment
        
    Returns:
        Filtered dictionary of tag counts
    """
    if drop_if_subset_of is None:
        drop_if_subset_of = {"Randomised Controlled Trial"}
    
    filtered = {}
    tags = set(tag_counts.keys())
    
    for tag, count in tag_counts.items():
        # Check if this tag is a subset of any other tag in the drop list
        is_redundant = any(
            tag != other and tag in other 
            for other in drop_if_subset_of 
            if other in tags
        )
        
        if not is_redundant:
            filtered[tag] = count
    
    return filtered
