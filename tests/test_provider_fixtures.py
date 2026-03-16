# Provider fixtures for benchmarks
"""
Item #54: Provider fixtures for benchmarks.

Creates fixtures that mock HuggingFace, LMSYS, and ArtificialAnalysis
API responses for predictable benchmark sync tests.
"""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest


class TestHuggingFaceFixtures:
    """HuggingFace API response fixtures."""

    @pytest.fixture
    def hf_response_success(self):
        """Successful HuggingFace API response."""
        return {
            "id": "meta-llama/Llama-2-7b",
            "sha": "abc123",
            "created_at": "2023-07-18T00:00:00Z",
            "tags": ["text-generation", "llama"],
            "downloads": 5000000,
            "likes": 10000,
            "card_data": {
                "model-index": [
                    {
                        "name": "Llama-2-7b",
                        "results": [
                            {
                                "task": {"type": "text-generation"},
                                "dataset": {"name": "mmlu"},
                                "metrics": [{"name": "accuracy", "value": 0.45}],
                            }
                        ],
                    }
                ]
            },
        }

    @pytest.fixture
    def hf_models_list(self):
        """List of HuggingFace models response."""
        return [
            {
                "id": "meta-llama/Llama-2-7b",
                "downloads": 5000000,
                "likes": 10000,
            },
            {
                "id": "mistralai/Mistral-7B-v0.1",
                "downloads": 3000000,
                "likes": 8000,
            },
        ]

    @pytest.fixture
    def hf_benchmark_data(self):
        """HuggingFace benchmark data for a model."""
        return {
            "mmlu": 0.45,
            "hellaswag": 0.62,
            "arc": 0.53,
            "winogrande": 0.68,
        }

    def test_hf_response_structure(self, hf_response_success):
        """Test HuggingFace response has expected structure."""
        assert "id" in hf_response_success
        assert "card_data" in hf_response_success


class TestLMSYSFixtures:
    """LMSYS Chatbot Arena fixtures."""

    @pytest.fixture
    def lmsys_leaderboard(self):
        """LMSYS leaderboard response."""
        return {
            "data": [
                {
                    "model": "gpt-4",
                    "elo": 1250,
                    "votes": 50000,
                },
                {
                    "model": "claude-3-opus",
                    "elo": 1240,
                    "votes": 45000,
                },
                {
                    "model": "llama-2-70b",
                    "elo": 1150,
                    "votes": 30000,
                },
            ],
            "last_updated": "2024-03-15T10:00:00Z",
        }

    @pytest.fixture
    def lmsys_model_stats(self):
        """LMSYS model statistics."""
        return {
            "gpt-4": {
                "elo": 1250,
                "confidence": 0.95,
                "wins": 30000,
                "losses": 15000,
            },
            "claude-3-opus": {
                "elo": 1240,
                "confidence": 0.94,
                "wins": 28000,
                "losses": 16000,
            },
        }

    def test_lmsys_response_structure(self, lmsys_leaderboard):
        """Test LMSYS response has expected structure."""
        assert "data" in lmsys_leaderboard
        assert len(lmsys_leaderboard["data"]) > 0
        assert "model" in lmsys_leaderboard["data"][0]


class TestArtificialAnalysisFixtures:
    """ArtificialAnalysis.ai fixtures."""

    @pytest.fixture
    def aa_api_response(self):
        """ArtificialAnalysis API response."""
        return {
            "models": [
                {
                    "id": "llama-2-70b",
                    "provider": "meta",
                    "indices": {
                        "intelligence": 0.65,
                        "coding": 0.58,
                        "math": 0.52,
                    },
                    "speed": {
                        "tokens_per_second": 45,
                        "time_to_first_token": 0.5,
                    },
                },
                {
                    "id": "gpt-4",
                    "provider": "openai",
                    "indices": {
                        "intelligence": 0.85,
                        "coding": 0.78,
                        "math": 0.75,
                    },
                    "speed": {
                        "tokens_per_second": 25,
                        "time_to_first_token": 0.3,
                    },
                },
            ],
            "benchmarks": {
                "mmlu_pro": {
                    "llama-2-70b": 0.63,
                    "gpt-4": 0.86,
                },
                "gpqa": {
                    "llama-2-70b": 0.35,
                    "gpt-4": 0.48,
                },
            },
        }

    @pytest.fixture
    def aa_model_mapping(self):
        """ArtificialAnalysis to Ollama model mapping."""
        return {
            "mappings": {
                "llama-2-70b": "llama2:70b",
                "gpt-4": None,  # Not available in Ollama
                "claude-3-opus": None,
            }
        }

    def test_aa_response_structure(self, aa_api_response):
        """Test ArtificialAnalysis response has expected structure."""
        assert "models" in aa_api_response
        assert "benchmarks" in aa_api_response


class TestBenchmarkSyncFixtures:
    """Combined benchmark sync fixtures."""

    @pytest.fixture
    def merged_benchmarks(self):
        """Merged benchmark data from multiple sources."""
        return {
            "llama-2-7b": {
                "mmlu": 0.45,  # From HuggingFace
                "lmsys_elo": 1150,  # From LMSYS
                "intelligence": 0.55,  # From ArtificialAnalysis
            },
            "mistral-7b": {
                "mmlu": 0.62,
                "lmsys_elo": 1180,
                "intelligence": 0.62,
            },
        }

    @pytest.fixture
    def sync_result_success(self):
        """Successful benchmark sync result."""
        return {
            "status": "success",
            "sources_synced": ["huggingface", "lmsys", "artificial_analysis"],
            "models_updated": 25,
            "models_added": 5,
            "errors": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @pytest.fixture
    def sync_result_partial_failure(self):
        """Partial failure sync result."""
        return {
            "status": "partial",
            "sources_synced": ["huggingface", "lmsys"],
            "models_updated": 20,
            "models_added": 0,
            "errors": ["artificial_analysis: API rate limit exceeded"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def test_merged_benchmarks_structure(self, merged_benchmarks):
        """Test merged benchmarks have expected structure."""
        for model, scores in merged_benchmarks.items():
            assert isinstance(scores, dict)
            assert len(scores) > 0


class TestProviderMocks:
    """Mock provider clients for testing."""

    @pytest.fixture
    def mock_hf_client(self, hf_models_list, hf_response_success):
        """Mock HuggingFace client."""
        client = AsyncMock()
        client.get_models.return_value = hf_models_list
        client.get_model_info.return_value = hf_response_success
        return client

    @pytest.fixture
    def mock_lmsys_client(self, lmsys_leaderboard):
        """Mock LMSYS client."""
        client = AsyncMock()
        client.get_leaderboard.return_value = lmsys_leaderboard
        return client

    @pytest.fixture
    def mock_aa_client(self, aa_api_response):
        """Mock ArtificialAnalysis client."""
        client = AsyncMock()
        client.get_models.return_value = aa_api_response["models"]
        client.get_benchmarks.return_value = aa_api_response["benchmarks"]
        return client

    @pytest.mark.asyncio
    async def test_mock_providers_return_expected_data(
        self, mock_hf_client, mock_lmsys_client, mock_aa_client
    ):
        """Test mock providers return expected fixture data."""
        hf_models = await mock_hf_client.get_models()
        assert len(hf_models) == 2

        lmsys_data = await mock_lmsys_client.get_leaderboard()
        assert len(lmsys_data["data"]) == 3

        aa_models = await mock_aa_client.get_models()
        assert len(aa_models) == 2
