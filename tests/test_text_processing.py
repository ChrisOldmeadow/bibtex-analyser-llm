""
Tests for text processing utilities.
"""

import pytest
from bibtex_analyzer.utils.text_processing import (
    normalize_tags,
    slugify,
    clean_text,
    collapse_tags,
    filter_redundant_tags
)


def test_normalize_tags():
    """Test tag normalization."""
    # Test basic normalization
    assert normalize_tags(["tag1", "TAG2", "Tag Three", "tag4"]) == ["Tag1", "Tag2", "Tag Three", "Tag4"]
    
    # Test with different styles
    assert normalize_tags(["TAG1", "tag2"], style="lower") == ["tag1", "tag2"]
    assert normalize_tags(["tag1", "tag2"], style="upper") == ["TAG1", "TAG2"]
    
    # Test with empty and whitespace tags
    assert normalize_tags(["", "  ", "tag1"]) == ["Tag1"]


def test_slugify():
    """Test slug generation."""
    assert slugify("Hello World") == "hello-world"
    assert slugify("Test 123") == "test-123"
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  extra  spaces  ") == "extra-spaces"
    assert slugify("Case-Sensitive") == "case-sensitive"


def test_clean_text():
    """Test text cleaning."""
    assert clean_text("  Hello   World  ") == "Hello World"
    assert clean_text("Test\nNew\tLine") == "Test New Line"
    assert clean_text("Special!@#chars$") == "Special chars"
    assert clean_text("") == ""


def test_collapse_tags():
    """Test tag collapsing with synonyms."""
    tags = ["RCT", "Randomised Controlled Trial", "Pragmatic Trial", "Something Else"]
    expected = {"Randomised Controlled Trial", "Something Else"}
    
    result = set(collapse_tags(tags))
    assert result == expected


def test_filter_redundant_tags():
    """Test filtering of redundant tags."""
    tag_counts = {
        "Randomised Controlled Trial": 10,
        "Pragmatic Trial": 5,
        "Something Else": 3
    }
    
    # Pragmatic Trial should be removed as it's a subset of Randomised Controlled Trial
    result = filter_redundant_tags(
        tag_counts,
        drop_if_subset_of={"Randomised Controlled Trial"}
    )
    
    assert "Pragmatic Trial" not in result
    assert "Randomised Controlled Trial" in result
    assert "Something Else" in result
    assert len(result) == 2
