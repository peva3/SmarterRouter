"""Tests for Ollama client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from router.client import GenerateResponse, ModelInfo, OllamaClient


class TestOllamaClient:
    """Test Ollama HTTP client."""

    @pytest.fixture
    def client(self):
        return OllamaClient(base_url="http://test:11434", timeout=30.0)

    @pytest.mark.asyncio
    async def test_list_models_success(self, client):
        """Test successful model listing."""
        mock_response = {
            "models": [
                {"name": "llama3", "size": 1000000000, "modified_at": "2024-01-01"},
                {"name": "mistral", "size": 2000000000, "modified_at": "2024-01-02"},
            ]
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            models = await client.list_models()

            assert len(models) == 2
            assert models[0].name == "llama3"
            assert models[1].name == "mistral"

    @pytest.mark.asyncio
    async def test_list_models_http_error(self, client):
        """Test handling of HTTP errors when listing models."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = httpx.HTTPError("Connection failed")
            models = await client.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful text generation."""
        mock_response = {
            "response": "Generated text",
            "done": True,
            "context": [1, 2, 3],
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.generate("llama3", "Test prompt")

            assert isinstance(result, GenerateResponse)
            assert result.response == "Generated text"
            assert result.done is True

    @pytest.mark.asyncio
    async def test_generate_http_error(self, client):
        """Test handling of HTTP errors during generation."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = httpx.HTTPError("Request failed")
            
            with pytest.raises(httpx.HTTPError):
                await client.generate("llama3", "Test prompt")

    @pytest.mark.asyncio
    async def test_chat_success(self, client):
        """Test successful chat completion."""
        mock_response = {
            "message": {"role": "assistant", "content": "Chat response"},
            "done": True,
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            messages = [{"role": "user", "content": "Hello"}]
            result = await client.chat("llama3", messages)

            assert result["message"]["content"] == "Chat response"


class TestGenerateResponse:
    """Test GenerateResponse dataclass."""

    def test_default_values(self):
        """Test default values for GenerateResponse."""
        response = GenerateResponse(response="Test", done=True)
        
        assert response.response == "Test"
        assert response.done is True
        assert response.context is None
        assert response.total_duration is None

    def test_full_values(self):
        """Test GenerateResponse with all fields."""
        response = GenerateResponse(
            response="Test",
            done=True,
            context=[1, 2, 3],
            total_duration=1000000,
            load_duration=100000,
            prompt_eval_count=10,
            eval_count=20,
        )
        
        assert response.context == [1, 2, 3]
        assert response.total_duration == 1000000
        assert response.prompt_eval_count == 10


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test ModelInfo creation."""
        info = ModelInfo(name="llama3", size=1000000000, modified_at="2024-01-01")
        
        assert info.name == "llama3"
        assert info.size == 1000000000
        assert info.modified_at == "2024-01-01"
