"""Tests for benchmark providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from router.providers.artificial_analysis import ArtificialAnalysisProvider
from router.providers.base import BenchmarkProvider
from router.providers.huggingface import HuggingFaceProvider
from router.providers.lmsys import LMSYSProvider


class TestBenchmarkProviderBase:
    """Test base provider interface."""

    def test_base_provider_is_abstract(self):
        """Ensure BenchmarkProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BenchmarkProvider()  # type: ignore


class TestHuggingFaceProvider:
    """Test HuggingFace leaderboard provider."""

    @pytest.fixture
    def provider(self):
        return HuggingFaceProvider()

    @pytest.mark.asyncio
    async def test_fetch_data_success(self, provider):
        """Test successful data fetch from HuggingFace via REST API."""
        mock_response = {
            "rows": [
                {
                    "row": {
                        "model_name": "Meta-Llama-3-8B",
                        "results": "{\"leaderboard\": {\"mmlu\": 66.0, \"humaneval\": 42.0, \"math\": 25.0}}"
                    }
                },
                {
                    "row": {
                        "model_name": "Mistral-7B-v0.2",
                        "results": "{\"leaderboard\": {\"mmlu\": 64.0, \"humaneval\": 40.0}}"
                    }
                },
            ]
        }

        with patch("router.providers.huggingface.httpx.AsyncClient") as mock_client_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response_obj)

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            mock_client_class.return_value = mock_context

            result = await provider.fetch_data(["llama3", "mistral"])

            assert len(result) == 2
            assert result[0]["ollama_name"] == "llama3"
            assert result[0]["mmlu"] == 66.0
            assert result[0]["humaneval"] == 42.0
            assert "reasoning_score" in result[0]
            assert "coding_score" in result[0]

    @pytest.mark.asyncio
    async def test_fetch_data_empty_response(self, provider):
        """Test handling of empty dataset."""
        mock_response = {"rows": []}

        with patch("router.providers.huggingface.httpx.AsyncClient") as mock_client_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response_obj)

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            mock_client_class.return_value = mock_context

            result = await provider.fetch_data(["llama3"])
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_data_exception_handling(self, provider):
        """Test graceful handling of HTTP errors."""
        with patch("router.providers.huggingface.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_client_instance = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("HTTP 500")
            mock_client_instance.get = AsyncMock(return_value=mock_response)

            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            mock_client_class.return_value = mock_context

            result = await provider.fetch_data(["llama3"])
            assert result == []

    def test_normalize_name(self, provider):
        """Test model name normalization."""
        assert provider._normalize_name("llama3:8b") == "llama3"
        assert provider._normalize_name("Llama-3-8B") == "llama38b"
        assert provider._normalize_name("mistral:7b-instruct") == "mistral"

    def test_match_model_direct(self, provider):
        """Test direct model matching."""
        ollama_models = ["llama3", "mistral"]
        ollama_base = {"llama3", "mistral"}

        result = provider._match_model("Meta-Llama-3-8B", ollama_models, ollama_base)
        assert result == "llama3"

    def test_match_model_fuzzy(self, provider):
        """Test fuzzy model matching."""
        ollama_models = ["custom-llama3-model"]
        ollama_base = {"customllama3model"}

        result = provider._match_model("Llama-3", ollama_models, ollama_base)
        assert result is not None

    def test_calculate_scores_complete(self, provider):
        """Test score calculation with all metrics."""
        data = {
            "mmlu": 70.0,
            "mmlu_pro": 65.0,
            "gpqa": 45.0,
            "math": 40.0,
            "humaneval": 50.0,
            "hellaswag": 80.0,
            "winogrande": 75.0,
            "truthfulqa": 60.0,
        }

        scores = provider._calculate_scores(data)

        assert 0 <= scores["reasoning_score"] <= 1
        assert 0 <= scores["coding_score"] <= 1
        assert 0 <= scores["general_score"] <= 1
        assert scores["coding_score"] > 0  # Should have coding score from humaneval

    def test_calculate_scores_partial(self, provider):
        """Test score calculation with partial metrics."""
        data = {"mmlu": 70.0}

        scores = provider._calculate_scores(data)

        assert scores["reasoning_score"] > 0
        assert scores["coding_score"] == 0  # No coding data
        assert scores["general_score"] > 0

    def test_extract_parameters(self, provider):
        """Test parameter extraction from model names."""
        assert provider._extract_parameters("Llama-3-8B") == "8B"
        assert provider._extract_parameters("Model-70b-chat") == "70B"
        assert provider._extract_parameters("Small-Model") is None

    def test_extract_scores_leaderboard(self, provider):
        """Test score extraction from leaderboard structure."""
        results = {
            "leaderboard": {
                "mmlu": 66.0,
                "humaneval": 42.0,
                "math": 25.0,
                "gpqa": 35.0,
            }
        }
        row = {"model_name": "test-model", "results": "{}"}

        scores = provider._extract_scores(results, row)

        assert scores["mmlu"] == 66.0
        assert scores["humaneval"] == 42.0
        assert scores["math"] == 25.0
        assert scores["gpqa"] == 35.0

    def test_extract_scores_nested_structure(self, provider):
        """Test score extraction from nested results JSON string."""
        results = {
            "leaderboard": {"mmlu": 70.0, "math": 45.0},
            "leaderboard_hellaswag": {"hellaswag": 80.0},
            "leaderboard_mmlu_pro": {"mmlu_pro": 65.0}
        }
        row = {"model_name": "test-model", "results": "{}"}

        scores = provider._extract_scores(results, row)

        assert scores["mmlu"] == 70.0
        assert scores["math"] == 45.0
        assert scores["hellaswag"] == 80.0
        assert scores["mmlu_pro"] == 65.0


class TestLMSYSProvider:
    """Test LMSYS Chatbot Arena provider."""

    @pytest.fixture
    def provider(self):
        return LMSYSProvider()

    @pytest.mark.asyncio
    async def test_fetch_data_success(self, provider):
        """Test successful Elo data fetch."""
        csv_data = """model,elo
Meta-Llama-3-8B,1200
Mistral-7B-v0.2,1150
"""

        with patch("router.providers.lmsys.httpx.AsyncClient") as mock_client_class:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = csv_data.encode()
            
            # Create async context manager mock
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            
            mock_client_class.return_value = mock_context_manager

            result = await provider.fetch_data(["llama3", "mistral"])

            # Should find at least one match
            assert len(result) >= 1
            if result:
                assert "elo_rating" in result[0]

    @pytest.mark.asyncio
    async def test_fetch_data_http_error(self, provider):
        """Test handling of HTTP errors."""
        with patch("router.providers.lmsys.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            
            mock_client_instance = MagicMock()
            mock_client_instance.get = MagicMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = MagicMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = MagicMock(return_value=None)

            result = await provider.fetch_data(["llama3"])
            assert result == []

    def test_process_dataframe(self, provider):
        """Test DataFrame processing."""
        df = pd.DataFrame({
            "model": ["Meta-Llama-3-8B", "Mistral-7B-v0.2"],
            "elo": [1200, 1150],
        })

        result = provider._process_dataframe(df, ["llama3", "mistral"])

        assert len(result) == 2
        assert all("ollama_name" in r for r in result)
        assert all("elo_rating" in r for r in result)


class TestArtificialAnalysisProvider:
    """Test Artificial Analysis provider."""

    @pytest.fixture
    def provider(self):
        return ArtificialAnalysisProvider()

    def test_name_property(self, provider):
        """Test provider name."""
        assert provider.name == "artificial_analysis"

    @pytest.mark.asyncio
    async def test_fetch_data_returns_empty(self, provider):
        """Test that placeholder returns empty list."""
        result = await provider.fetch_data(["llama3"])
        assert result == []
