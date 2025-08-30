"""Tests for deduplication functionality in BibtexProcessor."""
import pytest
from bibtex_analyzer.bibtex_processor import BibtexProcessor


def test_deduplication_by_doi():
    """Test that entries with the same DOI are deduplicated."""
    processor = BibtexProcessor()
    
    entries = [
        {
            'ID': 'entry1',
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'abstract': 'This is a test abstract',
            'staff_id': 'STAFF001'
        },
        {
            'ID': 'entry2',
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'abstract': 'This is a test abstract',
            'staff_id': 'STAFF002'
        },
        {
            'ID': 'entry3',
            'title': 'Different Paper',
            'doi': '10.5678/different',
            'abstract': 'Different abstract',
            'staff_id': 'STAFF003'
        }
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    assert len(deduplicated) == 2
    assert processor.deduplication_stats['total_entries'] == 3
    assert processor.deduplication_stats['unique_entries'] == 2
    assert processor.deduplication_stats['duplicates_removed'] == 1
    
    # Check that staff IDs were preserved
    test_paper = next(e for e in deduplicated if e.get('doi') == '10.1234/test')
    assert 'all_staff_ids' in test_paper
    assert set(test_paper['all_staff_ids']) == {'STAFF001', 'STAFF002'}


def test_deduplication_by_title():
    """Test that entries with the same normalized title are deduplicated."""
    processor = BibtexProcessor()
    
    entries = [
        {
            'ID': 'entry1',
            'title': 'Machine Learning for Data Analysis',
            'abstract': 'ML abstract',
            'staff_id': 'STAFF001'
        },
        {
            'ID': 'entry2',
            'title': 'MACHINE LEARNING FOR DATA ANALYSIS',  # Same title, different case
            'abstract': 'ML abstract',
            'staff_id': 'STAFF002'
        },
        {
            'ID': 'entry3',
            'title': 'Machine  Learning   for  Data  Analysis',  # Extra spaces
            'abstract': 'ML abstract',
            'staff_id': 'STAFF003'
        },
        {
            'ID': 'entry4',
            'title': 'Deep Learning for Image Recognition',
            'abstract': 'DL abstract',
            'staff_id': 'STAFF004'
        }
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    assert len(deduplicated) == 2
    assert processor.deduplication_stats['duplicates_removed'] == 2
    
    # Check that all staff IDs were preserved for the ML paper
    ml_paper = next(e for e in deduplicated if 'machine learning' in e['title'].lower())
    assert len(ml_paper['all_staff_ids']) == 3
    assert set(ml_paper['all_staff_ids']) == {'STAFF001', 'STAFF002', 'STAFF003'}


def test_deduplication_doi_priority():
    """Test that DOI takes priority over title for deduplication."""
    processor = BibtexProcessor()
    
    entries = [
        {
            'ID': 'entry1',
            'title': 'Original Title',
            'doi': '10.1234/test',
            'abstract': 'Abstract 1',
            'staff_id': 'STAFF001'
        },
        {
            'ID': 'entry2',
            'title': 'Different Title But Same DOI',
            'doi': '10.1234/test',
            'abstract': 'Abstract 2',
            'staff_id': 'STAFF002'
        }
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    assert len(deduplicated) == 1
    assert processor.deduplication_stats['duplicates_removed'] == 1


def test_deduplication_preserves_fields():
    """Test that deduplication preserves fields from duplicates when master is missing them."""
    processor = BibtexProcessor()
    
    entries = [
        {
            'ID': 'entry1',
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'staff_id': 'STAFF001'
            # Missing abstract and journal
        },
        {
            'ID': 'entry2',
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'abstract': 'This is the abstract',
            'journal': 'Test Journal',
            'year': '2023',
            'staff_id': 'STAFF002'
        }
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    assert len(deduplicated) == 1
    paper = deduplicated[0]
    
    # Check that missing fields were filled in
    assert paper['abstract'] == 'This is the abstract'
    assert paper['journal'] == 'Test Journal'
    assert paper['year'] == '2023'
    assert set(paper['all_staff_ids']) == {'STAFF001', 'STAFF002'}


def test_deduplication_no_staff():
    """Test deduplication when entries have no staff_id."""
    processor = BibtexProcessor()
    
    entries = [
        {
            'ID': 'entry1',
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'abstract': 'Abstract'
        },
        {
            'ID': 'entry2',
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'abstract': 'Abstract'
        }
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    assert len(deduplicated) == 1
    assert processor.deduplication_stats['duplicates_removed'] == 1
    # Should not have all_staff_ids if no staff_id was present
    assert 'all_staff_ids' not in deduplicated[0]


def test_deduplication_stats_tracking():
    """Test that deduplication statistics are properly tracked."""
    processor = BibtexProcessor()
    
    entries = [
        {'ID': f'entry{i}', 'title': f'Paper {i//2}', 'doi': f'10.1234/test{i//2}'}
        for i in range(6)  # Creates 3 unique papers, each with 2 duplicates
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    stats = processor.get_deduplication_stats()
    assert stats['total_entries'] == 6
    assert stats['unique_entries'] == 3
    assert stats['duplicates_removed'] == 3
    assert len(deduplicated) == 3


def test_deduplication_empty_entries():
    """Test deduplication with empty entry list."""
    processor = BibtexProcessor()
    
    deduplicated = processor.deduplicate_entries([])
    
    assert deduplicated == []
    assert processor.deduplication_stats['total_entries'] == 0
    assert processor.deduplication_stats['unique_entries'] == 0
    assert processor.deduplication_stats['duplicates_removed'] == 0


def test_deduplication_special_characters_in_title():
    """Test deduplication handles special characters in titles."""
    processor = BibtexProcessor()
    
    entries = [
        {
            'ID': 'entry1',
            'title': 'Machine Learning: A Comprehensive Study!',
            'abstract': 'ML study'
        },
        {
            'ID': 'entry2',
            'title': 'Machine Learning - A Comprehensive Study?',  # Different punctuation
            'abstract': 'ML study'
        }
    ]
    
    deduplicated = processor.deduplicate_entries(entries)
    
    # Should be deduplicated as titles normalize to same value
    assert len(deduplicated) == 1
    assert processor.deduplication_stats['duplicates_removed'] == 1