"""Tests for schema helper functions."""

import pytest

from router.schemas import (
    ChatCompletionRequest,
    close_unclosed_code_block,
    is_unclosed_code_block,
    sanitize_model_name,
)


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


class TestModelNameSanitization:
    """Test model-name sanitization utility and schema wiring."""

    def test_sanitize_model_name_trims_whitespace(self):
        assert sanitize_model_name("  llama3.2:1b  ") == "llama3.2:1b"

    def test_sanitize_model_name_rejects_unsafe_chars(self):
        with pytest.raises(ValueError):
            sanitize_model_name("../bad model")

    def test_chat_request_model_uses_sanitization(self):
        req = ChatCompletionRequest(
            model="  qwen3.5:4b  ",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.model == "qwen3.5:4b"


class TestMessageContentLength:
    """Test per-message content length enforcement (Item #27)."""

    def test_normal_content_accepted(self):
        """Short content should be accepted without issue."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        assert req.messages[0].content == "Hello, how are you?"

    def test_content_at_limit_accepted(self, monkeypatch):
        """Content exactly at the limit should be accepted."""
        monkeypatch.setattr("router.config.settings.max_message_content_length", 100)
        from router.schemas import ChatMessage
        msg = ChatMessage(role="user", content="x" * 100)
        assert len(msg.content) == 100

    def test_content_over_limit_rejected(self, monkeypatch):
        """Content over the limit should raise ValueError."""
        monkeypatch.setattr("router.config.settings.max_message_content_length", 50)
        from router.schemas import ChatMessage
        with pytest.raises(Exception, match="too long"):
            ChatMessage(role="user", content="x" * 51)

    def test_multimodal_text_part_over_limit_rejected(self, monkeypatch):
        """Text parts in multimodal content should also be checked."""
        monkeypatch.setattr("router.config.settings.max_message_content_length", 50)
        from router.schemas import ChatMessage
        with pytest.raises(Exception, match="too long"):
            ChatMessage(
                role="user",
                content=[{"type": "text", "text": "x" * 51}],
            )

    def test_multimodal_image_part_not_checked(self, monkeypatch):
        """Non-text parts should not be length-checked."""
        monkeypatch.setattr("router.config.settings.max_message_content_length", 50)
        from router.schemas import ChatMessage
        msg = ChatMessage(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 200}},
            ],
        )
        assert len(msg.content) == 1

    def test_none_content_accepted(self):
        """None content (e.g., assistant with tool_calls) should be accepted."""
        from router.schemas import ChatMessage
        msg = ChatMessage(role="assistant", content=None)
        assert msg.content is None
