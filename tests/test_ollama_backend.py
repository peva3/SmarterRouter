"""Tests for OllamaBackend."""

import json

import httpx
import pytest
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
                return_value=httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "llama3", "size": 1000000000, "modified_at": "2024-01-01"}
                        ]
                    },
                )
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
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {"content": "Hello!"},
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                )
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
            stream, latency = await backend.chat_streaming(
                "llama3", [{"role": "user", "content": "Hi"}]
            )

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
            mock_http.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(404))
            result = await backend.unload_model("nonexistent")
            assert result is True

    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model load."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/tags").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "llama3", "size": 1000000000, "modified_at": "2024-01-01"}
                        ]
                    },
                )
            )
            mock_http.post("http://localhost:11434/api/generate").mock(
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

    @pytest.mark.asyncio
    async def test_get_running_models(self):
        """Test getting running models with VRAM info."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/ps").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "llama3.2:1b",
                                "model": "llama3.2:1b",
                                "size": 1500000000,
                                "size_vram": 1200000000,
                                "digest": "abc123",
                            }
                        ]
                    },
                )
            )
            result = await backend.get_running_models()
            assert "llama3.2:1b" in result
            assert result["llama3.2:1b"]["vram_bytes"] == 1200000000
            assert result["llama3.2:1b"]["size_bytes"] == 1500000000

    @pytest.mark.asyncio
    async def test_get_model_vram_usage(self):
        """Test getting VRAM usage for a specific model."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/ps").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "llama3.2:1b",
                                "model": "llama3.2:1b",
                                "size": 1500000000,
                                "size_vram": 1200000000,
                                "digest": "abc123",
                            }
                        ]
                    },
                )
            )
            # Test exact match
            vram = await backend.get_model_vram_usage("llama3.2:1b")
            assert vram is not None
            assert abs(vram - 1.12) < 0.01  # 1200000000 / (1024**3) ≈ 1.12 GB

    @pytest.mark.asyncio
    async def test_get_model_vram_usage_not_loaded(self):
        """Test getting VRAM for a model that's not loaded."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/ps").mock(
                return_value=httpx.Response(200, json={"models": []})
            )
            vram = await backend.get_model_vram_usage("nonexistent-model")
            assert vram is None

    @pytest.mark.asyncio
    async def test_get_model_vram_usage_with_prefix(self):
        """Test getting VRAM with model prefix."""
        backend = OllamaBackend("http://localhost:11434", model_prefix="custom-")

        with respx.mock() as mock_http:
            mock_http.get("http://localhost:11434/api/ps").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "custom-llama3",
                                "model": "custom-llama3",
                                "size": 4000000000,
                                "size_vram": 3500000000,
                                "digest": "abc123",
                            }
                        ]
                    },
                )
            )
            # Should match with prefix applied
            vram = await backend.get_model_vram_usage("llama3")
            assert vram is not None
            assert abs(vram - 3.26) < 0.01  # 3500000000 / (1024**3) ≈ 3.26 GB

    @pytest.mark.asyncio
    async def test_chat_multimodal_non_streaming(self):
        """Test chat with multimodal messages transforms correctly for non-streaming."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "Image described"}})
            )
            
            # OpenAI-style multimodal message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}},
                    ]
                }
            ]
            
            await backend.chat("llava", messages, stream=False)
            
            # Check that the request was transformed to Ollama format
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            
            # Should have transformed messages
            assert len(body["messages"]) == 1
            msg = body["messages"][0]
            assert msg["role"] == "user"
            assert msg["content"] == "What's in this image?"
            assert "images" in msg
            assert len(msg["images"]) == 1
            # Should have stripped the data URL prefix
            assert msg["images"][0] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    @pytest.mark.asyncio
    async def test_chat_multimodal_streaming(self):
        """Test chat with multimodal messages transforms correctly for streaming."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"message": {"content": "Image described"}}\n{"done": true}\n',
                    headers={"content-type": "text/plain"},
                )
            )
            
            # OpenAI-style multimodal message with regular URL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                    ]
                }
            ]
            
            stream, latency = await backend.chat_streaming("llava", messages)
            
            # Consume the stream
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            
            # Check that the request was transformed to Ollama format
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            
            # Should have transformed messages
            assert len(body["messages"]) == 1
            msg = body["messages"][0]
            assert msg["role"] == "user"
            assert msg["content"] == "Describe this image"
            assert "images" in msg
            assert len(msg["images"]) == 1
            # Should have preserved the URL as-is
            assert msg["images"][0] == "http://example.com/image.jpg"

    @pytest.mark.asyncio
    async def test_chat_multimodal_mixed_content(self):
        """Test chat with mixed content types (text and multiple images)."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "Two images described"}})
            )
            
            # OpenAI-style multimodal message with multiple images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these images:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,first"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,second"}},
                    ]
                }
            ]
            
            await backend.chat("llava", messages, stream=False)
            
            # Check that the request was transformed to Ollama format
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            
            # Should have transformed messages
            assert len(body["messages"]) == 1
            msg = body["messages"][0]
            assert msg["role"] == "user"
            assert msg["content"] == "Compare these images:"
            assert "images" in msg
            assert len(msg["images"]) == 2
            # Should have stripped the data URL prefixes
            assert msg["images"][0] == "first"
            assert msg["images"][1] == "second"

    @pytest.mark.asyncio
    async def test_chat_regular_messages_unchanged(self):
        """Test that regular text-only messages are unchanged."""
        backend = OllamaBackend("http://localhost:11434")

        with respx.mock() as mock_http:
            mock_http.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "Regular response"}})
            )
            
            # Regular text-only message
            messages = [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
            
            await backend.chat("llama3", messages, stream=False)
            
            # Check that messages are unchanged
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            
            assert len(body["messages"]) == 2
            assert body["messages"][0] == {"role": "user", "content": "Hello, how are you?"}
            assert body["messages"][1] == {"role": "assistant", "content": "I'm doing well!"}
            # Should not have images field
            assert "images" not in body["messages"][0]
            assert "images" not in body["messages"][1]
