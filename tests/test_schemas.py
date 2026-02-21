"""Tests for schema helper functions."""

import pytest

from router.schemas import is_unclosed_code_block, close_unclosed_code_block


class TestCodeBlockHandling:
    """Test code block detection and closing."""

    def test_is_unclosed_code_block_false_for_empty(self):
        assert not is_unclosed_code_block("")

    def test_is_unclosed_code_block_false_for_closed(self):
        assert not is_unclosed_code_block("```\ncode\n```")

    def test_is_unclosed_code_block_true_for_unclosed(self):
        assert is_unclosed_code_block("```\ncode")

    def test_is_unclosed_code_block_multiple_blocks(self):
        # Two blocks, second unclosed
        assert is_unclosed_code_block("```\nfirst\n```\n```\nsecond")

    def test_is_unclosed_code_block_tilde_unclosed(self):
        assert is_unclosed_code_block("~~~\ncode")

    def test_is_unclosed_code_block_tilde_closed(self):
        assert not is_unclosed_code_block("~~~\ncode\n~~~")

    def test_is_unclosed_code_block_mixed_fences(self):
        # Backticks closed, tildes unclosed
        assert is_unclosed_code_block("```\ncode\n```\n~~~\nmore")
        # Tildes closed, backticks unclosed
        assert is_unclosed_code_block("~~~\ncode\n~~~\n```\nmore")

    def test_close_unclosed_code_block_adds_fence(self):
        content = "```\nprint('hello')"
        result = close_unclosed_code_block(content)
        assert result == "```\nprint('hello')\n```\n"

    def test_close_unclosed_code_block_tilde_adds_fence(self):
        content = "~~~\nprint('hello')"
        result = close_unclosed_code_block(content)
        assert result == "~~~\nprint('hello')\n~~~\n"

    def test_close_unclosed_code_block_no_change_for_closed(self):
        content = "```\ncode\n```"
        result = close_unclosed_code_block(content)
        assert result == content

    def test_close_unclosed_code_block_no_change_for_closed_tilde(self):
        content = "~~~\ncode\n~~~"
        result = close_unclosed_code_block(content)
        assert result == content

    def test_close_unclosed_code_block_handles_trailing_newline(self):
        content = "```\ncode\n"
        result = close_unclosed_code_block(content)
        assert result == "```\ncode\n```\n"

    def test_close_unclosed_code_block_handles_trailing_newline_tilde(self):
        content = "~~~\ncode\n"
        result = close_unclosed_code_block(content)
        assert result == "~~~\ncode\n~~~\n"

    def test_close_unclosed_code_block_empty_content(self):
        assert close_unclosed_code_block("") == ""
