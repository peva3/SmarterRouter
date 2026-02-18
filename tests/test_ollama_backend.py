"""Tests for OllamaBackend."""

import json
import pytest
import httpx
import respx

from router.backends.ollama import OllamaBackend


class TestOllamaBackend:
    """Test Ollama backend implementation."""

    def test_init_default_values(self):
        """Test initialization with defaults."""
        backend = OllamaBackend("http://localhost:11434")
        assert backend.base_url == "http://localhost:11434"
        assert backend.model_prefix == ""
        assert backend.timeout == 60.0
        assert backend.generation_timeout == 120.0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        backend = OllamaBackend(
            base_url="http://ollama:11434",
            model_prefix="custom/",
            timeout=30.0,
            generation_timeout=180.0,
        )
        assert backend.base_url == "http://ollama:11434"
        assert backend.model_prefix == "custom/"
        assert backend.timeout == 30.0
        assert backend.generation_timeout == 180.0

    def test_full_model_name_no_prefix(self):
        """Test _full_model_name without prefix."""
        backend = OllamaBackend("http://localhost:11434")
        assert backend._full_model_name("llama3") == "llama3"

    def test_full_model_name_with_prefix(self):
        """Test _full_model_name with prefix."""
        backend = OllamaBackend("http://localhost:11434", model_prefix="myorg/")
        assert backend._full_model_name("llama3") == "myorg/llama3"
        assert backend._full_model_name("myorg/llama3") == "myorg/llama3"  # Already prefixed

    def test_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base_url."""
        backend = OllamaBackend("http://localhost:11434/")
        assert backend.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test successful model listing."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/tags").mock(
                return_value=httpx.Response(200, json={
                    "models": [
                        {"name": "llama3", "size": 1000000000, "modified_at": "2024-01-01"}
                    ]
                })
            )
            models = await backend.list_models()
            assert len(models) == 1
            assert models[0].name == "llama3"
            assert models[0].size == 1000000000

    @pytest.mark.asyncio
    async def test_list_models_error_returns_empty(self):
        """Test that list_models returns empty list on error."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/tags").mock(
                side_effect=httpx.HTTPError("Connection failed")
            )
            models = await backend.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat completion."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={
                    "message": {"content": "Hello!"},
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                })
            )
            result = await backend.chat("llama3", [{"role": "user", "content": "Hi"}])
            assert result["message"]["content"] == "Hello!"
            assert result["prompt_eval_count"] == 10
            assert result["eval_count"] == 5

    @pytest.mark.asyncio
    async def test_chat_with_model_prefix(self):
        """Test chat applies model prefix."""
        backend = OllamaBackend("http://localhost:11434", model_prefix="myorg/")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "OK"}})
            )
            await backend.chat("llama3", [{"role": "user", "content": "test"}])
            
            # Check that prefixed model was sent
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["model"] == "myorg/llama3"

    @pytest.mark.asyncio
    async def test_chat_streaming(self):
        """Test streaming chat completion."""
        backend = OllamaBackend("http://localhost:11434")

        stream_data = [
            b'{"message": {"content": "Hello"}}\n',
            b'{"message": {"content": " World"}}\n',
            b'{"done": true}\n',
        ]

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b"".join(stream_data),
                    headers={"content-type": "text/plain"},
                )
            )
            stream, latency = await backend.chat_streaming("llama3", [{"role": "user", "content": "Hi"}])
            
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            
            assert len(chunks) >= 2
            assert chunks[0]["message"]["content"] == "Hello"
            assert chunks[1]["message"]["content"] == " World"
            assert latency >= 0

    @pytest.mark.asyncio
    async def test_unload_model_success(self):
        """Test successful model unload."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={})
            )
            result = await backend.unload_model("llama3")
            assert result is True

    @pytest.mark.asyncio
    async def test_unload_model_404_returns_true(self):
        """Test that 404 is treated as already unloaded."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(404)
            )
            result = await backend.unload_model("nonexistent")
            assert result is True

    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model load."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={})
            )
            result = await backend.load_model("llama3")
            assert result is True

    @pytest.mark.asyncio
    async def test_embed(self):
        """Test embedding generation."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/embed").mock(
                return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
            )
            result = await backend.embed("llama3", "test text")
            assert "embeddings" in result

    @pytest.mark.asyncio
    async def test_keep_alive_parameter(self):
        """Test keep_alive is passed correctly."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "OK"}})
            )
            await backend.chat("llama3", [{"role": "user", "content": "test"}], keep_alive=300)
            
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["keep_alive"] == 300
