"""
Tests for the TagGenerator class.
"""

import pytest
from unittest.mock import MagicMock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from bibtex_analyzer.tag_generator import TagGenerator

# Sample data for testing
SAMPLE_ENTRIES = [
    {"id": "key1", "title": "Title 1", "abstract": "This is a test abstract about machine learning."},
    {"id": "key2", "title": "Title 2", "abstract": "Another abstract about deep learning and AI."},
    {"id": "key3", "title": "Title 3", "abstract": "A third abstract about data science."}
]

# Mock response for tag generation
MOCK_TAG_RESPONSE = """Machine Learning, Deep Learning, Artificial Intelligence, Data Science, Neural Networks"""

# Mock response for tag assignment
MOCK_ASSIGNMENT_RESPONSE = """Machine Learning, Deep Learning"""


def create_mock_chat_completion(content: str) -> ChatCompletion:
    """Create a mock ChatCompletion object."""
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(finish_reason="stop", index=0, message=message)
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion"
    )


@patch('openai.OpenAI')
def test_generate_tags_for_abstracts(mock_openai):
    """Test generating tags for abstracts."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Configure the mock to return our test response
    mock_client.chat.completions.create.return_value = create_mock_chat_completion(MOCK_TAG_RESPONSE)
    
    # Initialize the tagger
    tagger = TagGenerator(api_key="test-key")
    
    # Test with default categories
    tags = tagger.generate_tags_for_abstracts(SAMPLE_ENTRIES, samples_per_category=2)
    
    # Verify the results
    assert isinstance(tags, set)
    assert len(tags) > 0
    assert all(isinstance(tag, str) for tag in tags)
    
    # Verify the API was called
    mock_client.chat.completions.create.assert_called()


@patch('openai.OpenAI')
def test_assign_tags_to_abstracts(mock_openai):
    """Test assigning tags to abstracts."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Configure the mock to return our test response
    mock_client.chat.completions.create.return_value = create_mock_chat_completion(MOCK_ASSIGNMENT_RESPONSE)
    
    # Initialize the tagger
    tagger = TagGenerator(api_key="test-key")
    
    # Test tag assignment
    tags = ["Machine Learning", "Deep Learning", "Artificial Intelligence", "Data Science"]
    entries = SAMPLE_ENTRIES.copy()
    
    tagged_entries = tagger.assign_tags_to_abstracts(entries, tags)
    
    # Verify the results
    assert len(tagged_entries) == len(entries)
    for entry in tagged_entries:
        assert "tags" in entry
        assert isinstance(entry["tags"], str)
        assert entry["tags"]  # Should not be empty
    
    # Verify the API was called for each abstract
    assert mock_client.chat.completions.create.call_count == len(entries)


@patch('builtins.input', return_value='y')
@patch('openai.OpenAI')
def test_interactive_tag_generation(mock_openai, mock_input):
    """Test interactive tag generation."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Configure the mock to return our test response
    mock_client.chat.completions.create.return_value = create_mock_chat_completion(MOCK_TAG_RESPONSE)
    
    # Initialize the tagger
    tagger = TagGenerator(api_key="test-key")
    
    # Test interactive tag generation
    tags = tagger.interactive_tag_generation(SAMPLE_ENTRIES, start_n=2, max_n=4, step=2)
    
    # Verify the results
    assert isinstance(tags, list)
    assert len(tags) > 0
    assert all(isinstance(tag, str) for tag in tags)
    
    # Verify the API was called
    mock_client.chat.completions.create.assert_called()


def test_tag_normalization():
    """Test that tags are properly normalized."""
    tagger = TagGenerator()
    
    # Test with various tag formats
    test_cases = [
        ("machine learning", "Machine Learning"),
        ("DEEP LEARNING", "Deep Learning"),
        ("  extra  spaces  ", "Extra Spaces"),
        ("mixed-CASE123", "Mixed-Case123")
    ]
    
    for input_tag, expected in test_cases:
        normalized = tagger._normalize_tag(input_tag)
        assert normalized == expected, f"Expected '{expected}' but got '{normalized}' for input '{input_tag}'"


@patch('openai.OpenAI')
def test_error_handling(mock_openai):
    """Test error handling in tag generation."""
    # Setup mock to raise an exception
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    # Initialize the tagger
    tagger = TagGenerator(api_key="test-key")
    
    # Test with error handling
    with pytest.raises(Exception):
        tagger.generate_tags_for_abstracts(SAMPLE_ENTRIES[:1])
    
    # Test with error handling in tag assignment
    entries = SAMPLE_ENTRIES.copy()
    tagged_entries = tagger.assign_tags_to_abstracts(entries, ["Test"])
    
    # Entries should still be returned, but with empty tags
    assert len(tagged_entries) == len(entries)
    for entry in tagged_entries:
        assert entry["tags"] == ""
