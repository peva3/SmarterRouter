"""Tests for LlamaCppBackend."""

import json
import pytest
import httpx
import respx

from router.backends.llama_cpp import LlamaCppBackend


class TestLlamaCppBackend:
    """Test llama.cpp backend implementation."""

    def test_init_default_values(self):
        backend = LlamaCppBackend("http://localhost:8080")
        assert backend.base_url == "http://localhost:8080"
        assert backend.model_prefix == ""
        assert backend.timeout == 60.0

    def test_init_custom_values(self):
        backend = LlamaCppBackend(
            base_url="http://llamacpp:8080",
            model_prefix="custom/",
            timeout=120.0,
        )
        assert backend.base_url == "http://llamacpp:8080"
        assert backend.model_prefix == "custom/"
        assert backend.timeout == 120.0

    def test_full_model_name_no_prefix(self):
        backend = LlamaCppBackend("http://localhost:8080")
        assert backend._full_model_name("llama3") == "llama3"

    def test_full_model_name_with_prefix(self):
        backend = LlamaCppBackend("http://localhost:8080", model_prefix="myorg/")
        assert backend._full_model_name("llama3") == "myorg/llama3"
        assert backend._full_model_name("myorg/llama3") == "myorg/llama3"

    def test_trailing_slash_stripped(self):
        backend = LlamaCppBackend("http://localhost:8080/")
        assert backend.base_url == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        backend = LlamaCppBackend("http://localhost:8080")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:8080/v1/models").mock(
                return_value=httpx.Response(200, json={
                    "data": [
                        {"id": "llama3", "size": 1000000000}
                    ]
                })
            )
            models = await backend.list_models()
            assert len(models) == 1
            assert models[0].name == "llama3"

    @pytest.mark.asyncio
    async def test_chat_returns_ollama_format(self):
        backend = LlamaCppBackend("http://localhost:8080")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8080/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
            )
            result = await backend.chat("llama3", [{"role": "user", "content": "Test"}])
            
            assert "message" in result
            assert result["message"]["content"] == "Test response"
            assert result["prompt_eval_count"] == 10
            assert result["eval_count"] == 5

    @pytest.mark.asyncio
    async def test_chat_empty_response(self):
        backend = LlamaCppBackend("http://localhost:8080")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8080/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": []})
            )
            result = await backend.chat("llama3", [{"role": "user", "content": "Test"}])
            assert result["message"]["content"] == ""

    @pytest.mark.asyncio
    async def test_chat_streaming_normalization(self):
        backend = LlamaCppBackend("http://localhost:8080")

        stream_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " World"}}]}\n',
            b'data: [DONE]\n',
        ]

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8080/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=b"".join(stream_chunks),
                    headers={"content-type": "text/plain"},
                )
            )
            stream, latency = await backend.chat_streaming("llama3", [{"role": "user", "content": "Hi"}])
            
            chunks = []
            async for chunk in stream:
                if "message" in chunk:
                    chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0]["message"]["content"] == "Hello"
            assert chunks[1]["message"]["content"] == " World"

    @pytest.mark.asyncio
    async def test_unload_model_not_supported(self):
        backend = LlamaCppBackend("http://localhost:8080")
        result = await backend.unload_model("llama3")
        assert result is False

    @pytest.mark.asyncio
    async def test_embed(self):
        backend = LlamaCppBackend("http://localhost:8080")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8080/v1/embeddings").mock(
                return_value=httpx.Response(200, json={
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
                })
            )
            result = await backend.embed("llama3", "test text")
            assert "data" in result

    @pytest.mark.asyncio
    async def test_embed_with_prefix(self):
        backend = LlamaCppBackend("http://localhost:8080", model_prefix="myorg/")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8080/v1/embeddings").mock(
                return_value=httpx.Response(200, json={"data": []})
            )
            await backend.embed("llama3", "test")
            
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["model"] == "myorg/llama3"

    @pytest.mark.asyncio
    async def test_model_prefix_in_chat(self):
        backend = LlamaCppBackend("http://localhost:8080", model_prefix="myorg/")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:8080/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                })
            )
            await backend.chat("llama3", [{"role": "user", "content": "test"}])
            
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["model"] == "myorg/llama3"
