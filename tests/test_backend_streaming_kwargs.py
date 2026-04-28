"""Tests for streaming backend keyword argument passthrough.

These tests verify that all backend implementations of chat_streaming()
accept **kwargs and forward them to the upstream API,
preventing a recurrence of the TypeError crash
where streaming requests with extra params (tools, temperature, etc.)
would fail because chat_streaming() did not accept **kwargs.
"""

import inspect
import json

import httpx
import pytest
import respx

from router.backends.ollama import OllamaBackend
from router.backends.openai import OpenAIBackend
from router.backends.llama_cpp import LlamaCppBackend
from router.backends.base import LLMBackend


# ---------------------------------------------------------------------------
# 1. Protocol conformance – all backends must match the Protocol signature
# ---------------------------------------------------------------------------

class TestChatStreamingProtocolConformance:
    """Verify all backends implement the required chat_streaming signature
    including **kwargs."""

    BACKENDS_TO_CHECK = [
        ("OllamaBackend", OllamaBackend("http://localhost:11434")),
        ("OpenAIBackend", OpenAIBackend("http://localhost:8000", api_key="test")),
        ("LlamaCppBackend", LlamaCppBackend("http://localhost:8080")),
    ]

    def _get_args(self, backend_instance) -> set[str]:
        import inspect
        sig = inspect.signature(backend_instance.chat_streaming)
        return {
            name for name, param in sig.parameters.items()
            if name != "self"
        }

    def test_all_backends_accept_kwargs(self):
        """Every backend's chat_streaming must have **kwargs."""
        for name, backend in self.BACKENDS_TO_CHECK:
            sig = inspect.signature(backend.chat_streaming)
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            assert has_kwargs, (
                f"{name}.chat_streaming() is missing **kwargs — "
                "extra params like tools, temperature, etc. will crash the request"
            )

    def test_all_backends_have_minimum_required_params(self):
        """Every backend's chat_streaming must at least accept
        model, messages, keep_alive, and **kwargs."""
        for name, backend in self.BACKENDS_TO_CHECK:
            sig = inspect.signature(backend.chat_streaming)
            params = dict(sig.parameters)
            assert "model" in params
            assert "messages" in params
            assert "keep_alive" in params or "**kwargs" in str(sig)

    def test_protocol_declares_kwargs(self):
        """The LLMBackend Protocol itself must declare **kwargs."""
        sig = inspect.signature(LLMBackend.chat_streaming)
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert has_kwargs, (
            "LLMBackend Protocol chat_streaming() is missing **kwargs"
        )

    def test_all_backends_match_protocol(self):
        """All backends should be valid implementations of the Protocol.
        This is checked by issubclass() — Protocol.
        Since the Protocol uses @runtime_checkable,
        isinstance() checks the method signatures at runtime."""
        # Manual check: every backend should accept at minimum the Protocol params
        protocol_sig = inspect.signature(LLMBackend.chat_streaming)
        protocol_params = set(protocol_sig.parameters.keys()) - {"self"}

        for name, backend in self.BACKENDS_TO_CHECK:
            impl_sig = inspect.signature(backend.chat_streaming)
            impl_params = set(impl_sig.parameters.keys()) - {"self"}

            # The implementation must accept all Protocol params
            for pname in protocol_params:
                assert pname in impl_params, (
                    f"{name}.chat_streaming() missing required param '{pname}'"
                )


# ---------------------------------------------------------------------------
# 2. Extra kwargs are forwarded to the upstream API payload
# ---------------------------------------------------------------------------

class TestChatStreamingKwargsPassthrough:
    """Verify that extra keyword arguments are actually sent in the
    outgoing HTTP request body."""

    @pytest.mark.asyncio
    async def test_ollama_streaming_forwards_tools(self):
        """OllamaBackend.chat_streaming() must include tools in the payload."""
        backend = OllamaBackend("http://localhost:11434")
        tools_def = [{"type": "function", "function": {"name": "get_weather"}}]

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"message":{"content":"ok"},"done":true}\n',
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "llama3",
                [{"role": "user", "content": "Hi"}],
                tools=tools_def,
            )
            async for _ in stream:
                pass

            request = route.calls.last.request
            body = json.loads(request.content)

            assert "tools" in body, "tools parameter was dropped — not in outgoing payload"
            assert body["tools"] == tools_def

    @pytest.mark.asyncio
    async def test_ollama_streaming_forwards_temperature(self):
        """OllamaBackend.chat_streaming() must forward temperature."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"message":{"content":"ok"},"done":true}\n',
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "llama3",
                [{"role": "user", "content": "Hi"}],
                temperature=0.7,
                top_p=0.9,
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            assert body.get("temperature") == 0.7
            assert body.get("top_p") == 0.9

    @pytest.mark.asyncio
    async def test_ollama_streaming_forwards_tool_choice(self):
        """OllamaBackend.chat_streaming() must forward tool_choice."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"message":{"content":"ok"},"done":true}\n',
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "llama3",
                [{"role": "user", "content": "Hi"}],
                tool_choice="any",
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            assert body.get("tool_choice") == "any"

    @pytest.mark.asyncio
    async def test_openai_streaming_forwards_tools(self):
        """OpenAIBackend.chat_streaming() must include tools in the payload."""
        backend = OpenAIBackend("http://localhost:8000", api_key="test-key")
        tools_def = [{"type": "function", "function": {"name": "search"}}]

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:8000/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=b"data: [DONE]\n",
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "gpt-4",
                [{"role": "user", "content": "Hi"}],
                tools=tools_def,
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            assert "tools" in body, "tools parameter was dropped in OpenAI streaming"
            assert body["tools"] == tools_def

    @pytest.mark.asyncio
    async def test_llamacpp_streaming_forwards_tools(self):
        """LlamaCppBackend.chat_streaming() must include tools in the payload."""
        backend = LlamaCppBackend("http://localhost:8080")
        tools_def = [{"type": "function", "function": {"name": "calc"}}]

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:8080/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=b"data: [DONE]\n",
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "llama3",
                [{"role": "user", "content": "Hi"}],
                tools=tools_def,
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            assert "tools" in body, "tools parameter was dropped in llama.cpp streaming"
            assert body["tools"] == tools_def


# ---------------------------------------------------------------------------
# 3. Multiple extra kwargs forwarded simultaneously
# ---------------------------------------------------------------------------

class TestChatStreamingMultipleKwargs:
    """Test that batched kwargs all arrive in the payload."""

    @pytest.mark.asyncio
    async def test_multiple_kwargs_in_ollama_streaming(self):
        """Multiple extra kwargs forwarded to Ollama streaming endpoint."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"message":{"content":"ok"},"done":true}\n',
                    headers={"content-type": "text/plain"},
                )
            )

            extra = {
                "temperature": 0.3,
                "top_p": 0.95,
                "tools": [{"type": "function", "function": {"name": "x"}}],
                "tool_choice": "auto",
                "seed": 42,
                "max_tokens": 500,
            }

            stream, _ = await backend.chat_streaming(
                "llama3",
                [{"role": "user", "content": "Hi"}],
                **extra,
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            for key, value in extra.items():
                assert key in body, f"{key} missing from payload"
                assert body[key] == value, f"{key} has wrong value"


# ---------------------------------------------------------------------------
# 4. Extra kwargs do NOT interfere with required params
# ---------------------------------------------------------------------------

class TestChatStreamingRequiredParamsPreserved:
    """Required params (model, messages, stream) must be in every outgoing
    request even when extra kwargs are passed."""

    @pytest.mark.asyncio
    async def test_required_params_remain_in_ollama(self):
        """model, messages, stream remain when extra kwargs present."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"message":{"content":"ok"},"done":true}\n',
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "llama3",
                [{"role": "user", "content": "Hi"}],
                extra_param="should_not_break",
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            assert body["model"] == "llama3"
            assert body["messages"] == [{"role": "user", "content": "Hi"}]
            assert body["stream"] is True

    @pytest.mark.asyncio
    async def test_required_params_remain_in_openai(self):
        """model, messages, stream remain when extra kwargs present for OpenAI."""
        backend = OpenAIBackend("http://localhost:8000", api_key="test-key")

        with respx.mock() as mock_http:
            route = mock_http.post("http://localhost:8000/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=b"data: [DONE]\n",
                    headers={"content-type": "text/plain"},
                )
            )

            stream, _ = await backend.chat_streaming(
                "gpt-4",
                [{"role": "user", "content": "Hello"}],
                temperature=0.5,
            )
            async for _ in stream:
                pass

            body = json.loads(route.calls.last.request.content)
            assert body["model"] == "gpt-4"
            assert body["messages"] == [{"role": "user", "content": "Hello"}]
            assert body["stream"] is True
