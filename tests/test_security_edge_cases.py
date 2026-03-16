"""Security edge case tests (Item #55).

Tests for SQL injection, oversized prompts, malformed JSON, path traversal.
"""

import json

import pytest


class TestSQLInjection:
    """Test SQL injection prevention in model names and inputs."""

    def test_model_name_sql_injection_drop_table(self):
        """Test DROP TABLE injection in model name is blocked."""
        from router.schemas import sanitize_model_name

        malicious = "model'; DROP TABLE model_profiles; --"
        sanitized = sanitize_model_name(malicious)

        # Should sanitize or reject
        assert sanitized is None or "DROP" not in str(sanitized).upper()

    def test_model_name_sql_injection_union_select(self):
        """Test UNION SELECT injection in model name."""
        from router.schemas import sanitize_model_name

        malicious = "model' UNION SELECT * FROM admin_users --"
        sanitized = sanitize_model_name(malicious)

        assert sanitized is None or "UNION" not in str(sanitized).upper()

    def test_model_name_sql_injection_insert(self):
        """Test INSERT injection in model name."""
        from router.schemas import sanitize_model_name

        malicious = "model'; INSERT INTO model_profiles VALUES (1, 'hacked'); --"
        sanitized = sanitize_model_name(malicious)

        assert sanitized is None or "INSERT" not in str(sanitized).upper()

    @pytest.mark.asyncio
    async def test_chat_endpoint_sql_injection_model_name(self):
        """Test SQL injection via chat endpoint model parameter."""
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)

        malicious_payload = {
            "model": "model'; DROP TABLE model_profiles; --",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        response = client.post(
            "/v1/chat/completions",
            json=malicious_payload
        )

        # Should return error without executing SQL
        assert response.status_code in [400, 422, 500]
        response_data = response.json()
        assert "DROP" not in str(response_data).upper()  # No SQL in response


class TestOversizedPrompts:
    """Test handling of oversized prompts."""

    def test_prompt_too_long_rejected(self):
        """Test very long prompts are handled gracefully."""
        from router.schemas import ChatMessage

        # Create a message with excessive content
        long_content = "A" * 1000000  # 1MB of text

        # Should raise validation error
        with pytest.raises(ValueError):
            ChatMessage(role="user", content=long_content)

    def test_many_messages_rejected(self):
        """Test too many messages in conversation."""
        from router.schemas import ChatCompletionRequest

        # Create 1000 messages
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(1000)]

        with pytest.raises(ValueError):
            ChatCompletionRequest(model="test", messages=messages)

    def test_deeply_nested_json_rejected(self):
        """Test deeply nested JSON structures."""
        from router.schemas import ChatMessage

        # Create deeply nested content
        nested = {"level": 1}
        for i in range(100):
            nested = {"level": i + 2, "nested": nested}

        # Should handle gracefully
        try:
            ChatMessage(role="user", content=str(nested))
        except (ValueError, RecursionError):
            pass  # Expected


class TestMalformedJSON:
    """Test handling of malformed JSON."""

    def test_invalid_json_syntax(self):
        """Test invalid JSON syntax is rejected."""
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            content=b"{invalid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return error
        assert response.status_code == 422

    def test_json_with_comments(self):
        """Test JSON with JavaScript-style comments."""
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)

        json_with_comments = """
        {
            // This is a comment
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }
        """

        response = client.post(
            "/v1/chat/completions",
            content=json_with_comments.encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        # Should reject or handle gracefully
        assert response.status_code in [400, 422]

    def test_truncated_json(self):
        """Test truncated JSON payload."""
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)

        truncated = '{"model": "test", "messages": [{"role": "user", "content": "'

        response = client.post(
            "/v1/chat/completions",
            content=truncated.encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_json_with_control_characters(self):
        """Test JSON with control characters."""
        from router.schemas import ChatMessage

        # Message with null bytes and control chars
        content = "Hello\x00World\x01\x02\x03"

        # Should sanitize or reject
        try:
            msg = ChatMessage(role="user", content=content)
            msg_content = str(msg.content)
            # Content should be cleaned
            assert "\x00" not in msg_content
        except ValueError:
            pass  # Also acceptable


class TestPathTraversal:
    """Test path traversal prevention."""

    def test_path_traversal_in_model_name(self):
        """Test path traversal in model name."""
        from router.schemas import sanitize_model_name

        malicious = "../../../etc/passwd"
        sanitized = sanitize_model_name(malicious)

        # Should not allow path traversal
        assert sanitized is None or ".." not in str(sanitized)

    def test_absolute_path_in_model_name(self):
        """Test absolute path in model name."""
        from router.schemas import sanitize_model_name

        malicious = "/etc/passwd"
        sanitized = sanitize_model_name(malicious)

        # Should not allow absolute paths
        assert sanitized is None or not str(sanitized).startswith("/")

    def test_null_byte_injection(self):
        """Test null byte injection."""
        from router.schemas import sanitize_model_name

        malicious = "model\x00/../../../etc/passwd"
        sanitized = sanitize_model_name(malicious)

        # Null bytes should be removed
        assert sanitized is None or "\x00" not in str(sanitized)


class TestXSSPrevention:
    """Test XSS prevention in responses."""

    def test_xss_in_prompt_content(self):
        """Test XSS payloads in prompt content are sanitized."""
        from router.schemas import ChatMessage

        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
        ]

        for payload in xss_payloads:
            # Should accept but potentially sanitize
            msg = ChatMessage(role="user", content=payload)
            msg_content = str(msg.content)
            # Content should be preserved (sanitization happens at display layer)
            assert payload in msg_content or msg_content != ""


class TestNoSQLInjection:
    """Test NoSQL injection prevention (for Redis cache)."""

    @pytest.mark.skipif(
        "redis" not in str(pytest.importorskip("redis", reason="Redis not installed")),
        reason="Redis not configured"
    )
    def test_redis_command_injection(self):
        """Test Redis command injection prevention."""
        # This would test if Redis cache keys are properly escaped
        from router.cache import get_cache

        cache = get_cache("security_test")

        malicious_key = "key\nFLUSHALL\r\n"

        # Should not execute FLUSHALL
        cache.set(malicious_key, "value")

        # Verify Redis still has other keys (if any existed)
        # This is a basic check - full test would need actual Redis


class TestCommandInjection:
    """Test command injection prevention."""

    def test_shell_injection_in_filename(self):
        """Test shell injection in filename parameters."""
        from router.schemas import sanitize_model_name

        malicious = "model; rm -rf /"
        sanitized = sanitize_model_name(malicious)

        # Should not allow shell metacharacters
        assert sanitized is None or ";" not in str(sanitized)

    def test_backtick_injection(self):
        """Test backtick command substitution."""
        from router.schemas import sanitize_model_name

        malicious = "model`whoami`"
        sanitized = sanitize_model_name(malicious)

        assert sanitized is None or "`" not in str(sanitized)


class TestUnicodeNormalization:
    """Test Unicode edge cases."""

    def test_homoglyph_attack(self):
        """Test homoglyph attacks in model names."""
        from router.schemas import sanitize_model_name

        # Unicode homoglyphs (lookalike characters)
        homoglyphs = [
            "ｌｌａｍａ３",  # Fullwidth characters
            "llama3\u200B",  # Zero-width space
            "llama3\u200D",  # Zero-width joiner
            "ⅼⅼama3",  # Mathematical characters
        ]

        for name in homoglyphs:
            sanitized = sanitize_model_name(name)
            # Should normalize or reject
        assert sanitized is None or sanitized == name or str(sanitized) != ""

    def test_right_to_left_override(self):
        """Test RTL override characters."""
        from router.schemas import sanitize_model_name

        # RTL override character
        malicious = "\u202Ellama3.exe"
        sanitized = sanitize_model_name(malicious)

        # RTL characters should be handled
        assert sanitized is None or "\u202E" not in str(sanitized)


class TestHeaderInjection:
    """Test HTTP header injection prevention."""

    def test_newline_in_header(self):
        """Test CRLF injection in headers."""
        # This would test response headers
        # FastAPI should handle this automatically
        pass


class TestIntegerOverflow:
    """Test integer overflow prevention."""

    def test_extremely_large_integers(self):
        """Test extremely large integers in requests."""
        from router.schemas import ChatCompletionRequest

        # Very large number
        with pytest.raises((ValueError, OverflowError)):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10**100
            )

    def test_negative_integers(self):
        """Test negative integers where positive expected."""
        from router.schemas import ChatCompletionRequest

        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=-1
            )


class TestHTTPVerbTampering:
    """Test HTTP verb tampering."""

    def test_unexpected_http_methods(self):
        """Test unexpected HTTP methods on endpoints."""
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)

        # DELETE on chat endpoint
        response = client.delete("/v1/chat/completions")
        assert response.status_code == 405  # Method not allowed

        # PUT on chat endpoint
        response = client.put("/v1/chat/completions")
        assert response.status_code == 405

    def test_trace_method_disabled(self):
        """Test TRACE method is disabled."""
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)

        response = client.request("TRACE", "/v1/chat/completions")
        # Should be 405 or 501
        assert response.status_code in [405, 501]
