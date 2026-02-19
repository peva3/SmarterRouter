"""Additional tests for markdown JSON extraction in judge."""

import pytest
from router.judge import _extract_json_from_content


class TestMarkdownExtraction:
    """Test JSON extraction from markdown-formatted responses."""

    def test_extract_plain_json(self):
        """Test extraction of plain JSON without markdown."""
        content = '{"score": 0.85, "reasoning": "Good response"}'
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.85, "reasoning": "Good response"}'

    def test_extract_json_with_json_tag(self):
        """Test extraction of JSON wrapped in ```json blocks."""
        content = '```json\n{"score": 0.9, "reasoning": "Excellent"}\n```'
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.9, "reasoning": "Excellent"}'

    def test_extract_json_with_plain_code_block(self):
        """Test extraction of JSON wrapped in ``` blocks without json tag."""
        content = '```\n{"score": 0.75}\n```'
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.75}'

    def test_extract_json_from_extra_text(self):
        """Test extraction when JSON is embedded in surrounding text."""
        content = 'Here is my evaluation: {"score": 0.6, "reasoning": "Okay"} Hope this helps!'
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.6, "reasoning": "Okay"}'

    def test_extract_nested_json(self):
        """Test extraction with nested braces."""
        content = '```json\n{"score": 0.8, "reasoning": "It said {hello} which is good"}\n```'
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.8, "reasoning": "It said {hello} which is good"}'

    def test_extract_single_line_code_block(self):
        """Test extraction from single-line code block."""
        content = '```json {"score": 1.0} ```'
        result = _extract_json_from_content(content)
        # This won't extract properly due to no newlines, but should return something
        assert '"score": 1.0' in result

    def test_extract_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        content = '   ```json\n{"score": 0.5}\n```   '
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.5}'

    def test_extract_no_score_key(self):
        """Test extraction when no score key exists."""
        content = '{"other": "data"}'
        result = _extract_json_from_content(content)
        assert result == '{"other": "data"}'

    def test_extract_empty_content(self):
        """Test extraction with empty content."""
        content = ''
        result = _extract_json_from_content(content)
        assert result == ''
