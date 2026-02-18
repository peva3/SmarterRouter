"""Tests for OpenAIBackend."""

import json
import pytest
import httpx
import respx

from router.backends.openai import OpenAIBackend


class TestOpenAIBackend:
    """Test OpenAI-compatible backend implementation."""

    def test_init_default_values(self):
        backend = OpenAIBackend("https://api.openai.com/v1", api_key="sk-test")
        assert backend.base_url == "https://api.openai.com/v1"
        assert backend.api_key == "sk-test"
        assert backend.model_prefix == ""
        assert backend.timeout == 120.0

    def test_init_custom_values(self):
        backend = OpenAIBackend(
            base_url="http://localhost:8000/v1",
            api_key="my-key",
            model_prefix="custom/",
            timeout=180.0,
        )
        assert backend.base_url == "http://localhost:8000/v1"
        assert backend.api_key == "my-key"
        assert backend.model_prefix == "custom/"
        assert backend.timeout == 180.0

    def test_full_model_name_no_prefix(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="test")
        assert backend._full_model_name("gpt-4") == "gpt-4"

    def test_full_model_name_with_prefix(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="test", model_prefix="org/")
        assert backend._full_model_name("gpt-4") == "org/gpt-4"
        assert backend._full_model_name("org/gpt-4") == "org/gpt-4"

    def test_trailing_slash_stripped(self):
        backend = OpenAIBackend("http://localhost:8000/v1/", api_key="test")
        assert backend.base_url == "http://localhost:8000/v1"

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:8000/v1/models").mock(
                return_value=httpx.Response(200, json={
                    "data": [
                        {"id": "gpt-4", "size": 1000000000}
                    ]
                })
            )
            models = await backend.list_models()
            assert len(models) == 1
            assert models[0].name == "gpt-4"

    @pytest.mark.asyncio
    async def test_request_includes_auth_header(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
            )
            await backend.chat("gpt-4", [{"role": "user", "content": "test"}])
            
            request = mock_http.calls.last.request
            assert request.headers["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_chat_returns_ollama_format(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
            )
            result = await backend.chat("gpt-4", [{"role": "user", "content": "Test"}])
            
            assert "message" in result
            assert result["message"]["content"] == "Test response"
            assert result["prompt_eval_count"] == 10
            assert result["eval_count"] == 5

    @pytest.mark.asyncio
    async def test_chat_with_model_prefix(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test", model_prefix="myorg/")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
            )
            await backend.chat("gpt-4", [{"role": "user", "content": "test"}])
            
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["model"] == "myorg/gpt-4"

    @pytest.mark.asyncio
    async def test_chat_streaming_normalization(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        stream_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " World"}}]}\n',
            b'data: [DONE]\n',
        ]

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=b"".join(stream_chunks),
                    headers={"content-type": "text/plain"},
                )
            )
            stream, latency = await backend.chat_streaming("gpt-4", [{"role": "user", "content": "Hi"}])
            
            chunks = []
            async for chunk in stream:
                if "message" in chunk:
                    chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0]["message"]["content"] == "Hello"
            assert chunks[1]["message"]["content"] == " World"

    @pytest.mark.asyncio
    async def test_chat_streaming_auth_header(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n',
                )
            )
            stream, latency = await backend.chat_streaming("gpt-4", [{"role": "user", "content": "test"}])
            
            # Consume the stream to trigger the HTTP request
            async for _ in stream:
                pass
            
            request = mock_http.calls.last.request
            assert request.headers["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_unload_model_not_supported(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")
        result = await backend.unload_model("gpt-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_embed(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/embeddings").mock(
                return_value=httpx.Response(200, json={
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
                })
            )
            result = await backend.embed("text-embedding-ada-002", "test text")
            assert "data" in result

    @pytest.mark.asyncio
    async def test_chat_empty_response(self):
        backend = OpenAIBackend("http://localhost:8000/v1", api_key="sk-test")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": []})
            )
            result = await backend.chat("gpt-4", [{"role": "user", "content": "Test"}])
            assert result["message"]["content"] == ""
