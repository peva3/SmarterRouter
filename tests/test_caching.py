import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from router.benchmark_db import (
    get_all_benchmarks,
    get_benchmarks_for_models,
    invalidate_benchmarks_cache,
    invalidate_profiles_cache,
    invalidate_all_caches,
)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before each test."""
    invalidate_all_caches()
    yield
    invalidate_all_caches()


class TestBenchmarkCaching:
    """Tests for benchmark and profile caching."""

    @patch("router.benchmark_db.get_session")
    def test_get_all_benchmarks_caches_result(self, mock_get_session):
        """Test that get_all_benchmarks caches results."""
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_benchmark = MagicMock()
        mock_benchmark.ollama_name = "test-model"
        mock_benchmark.reasoning_score = 0.9
        mock_benchmark.coding_score = 0.8
        mock_benchmark.general_score = 0.85
        mock_benchmark.elo_rating = 1200
        mock_benchmark.throughput = 100
        mock_benchmark.parameters = "7B"
        mock_benchmark.mmlu = 0.7
        mock_benchmark.humaneval = 0.65

        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            mock_benchmark
        ]

        result1 = get_all_benchmarks()
        result2 = get_all_benchmarks()

        assert len(result1) == 1
        assert result1 == result2
        assert mock_session.execute.call_count == 1

    @patch("router.benchmark_db.get_session")
    def test_get_benchmarks_for_models_caches_result(self, mock_get_session):
        """Test that get_benchmarks_for_models caches results."""
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_benchmark = MagicMock()
        mock_benchmark.ollama_name = "llama3"
        mock_benchmark.reasoning_score = 0.9
        mock_benchmark.coding_score = 0.8
        mock_benchmark.general_score = 0.85
        mock_benchmark.elo_rating = 1200

        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            mock_benchmark
        ]

        model_names = ["llama3", "codellama"]
        result1 = get_benchmarks_for_models(model_names)
        result2 = get_benchmarks_for_models(model_names)

        assert len(result1) == 1
        assert result1 == result2
        assert mock_session.execute.call_count == 1

    @patch("router.benchmark_db.get_session")
    def test_get_benchmarks_for_models_different_models(self, mock_get_session):
        """Test that cache is not shared between different model name sets."""
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_benchmark = MagicMock()
        mock_benchmark.ollama_name = "llama3"
        mock_benchmark.reasoning_score = 0.9
        mock_benchmark.coding_score = 0.8
        mock_benchmark.general_score = 0.85
        mock_benchmark.elo_rating = 1200

        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            mock_benchmark
        ]

        result1 = get_benchmarks_for_models(["llama3"])
        assert len(result1) == 1

        result2 = get_benchmarks_for_models(["codellama"])
        assert len(result2) == 1

        assert mock_session.execute.call_count == 2

    def test_invalidate_benchmarks_cache(self):
        """Test that invalidate_benchmarks_cache clears the cache."""
        with patch("router.benchmark_db.get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

            mock_benchmark = MagicMock()
            mock_benchmark.ollama_name = "test-model"
            mock_benchmark.reasoning_score = 0.9
            mock_benchmark.coding_score = 0.8
            mock_benchmark.general_score = 0.85
            mock_benchmark.elo_rating = 1200
            mock_benchmark.throughput = 100
            mock_benchmark.parameters = "7B"
            mock_benchmark.mmlu = 0.7
            mock_benchmark.humaneval = 0.65

            mock_session.execute.return_value.scalars.return_value.all.return_value = [
                mock_benchmark
            ]

            result1 = get_all_benchmarks()
            invalidate_benchmarks_cache()
            result2 = get_all_benchmarks()

            assert len(result1) == 1
            assert len(result2) == 1
            assert mock_session.execute.call_count == 2


class TestRouterEngineCaching:
    """Tests for RouterEngine cache warming and invalidation."""

    def test_warmup_caches(self):
        """Test that warmup_caches pre-warms the caches."""
        from router.router import RouterEngine

        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[])

        engine = RouterEngine(
            client=mock_backend,
            cache_enabled=False,
        )

        with patch.object(engine, "_get_all_profiles") as mock_profiles:
            with patch("router.benchmark_db.get_benchmarks_for_models") as mock_benchmarks:
                engine.warmup_caches(["llama3", "codellama"])

                mock_profiles.assert_called_once()
                mock_benchmarks.assert_called_once_with(["llama3", "codellama"])

    def test_warmup_caches_without_model_names(self):
        """Test that warmup_caches works without model names."""
        from router.router import RouterEngine

        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[])

        engine = RouterEngine(
            client=mock_backend,
            cache_enabled=False,
        )

        with patch.object(engine, "_get_all_profiles") as mock_profiles:
            with patch("router.benchmark_db.get_benchmarks_for_models") as mock_benchmarks:
                engine.warmup_caches()

                mock_profiles.assert_called_once()
                mock_benchmarks.assert_not_called()

    def test_invalidate_caches(self):
        """Test that invalidate_caches calls invalidate_all_caches."""
        from router.router import RouterEngine

        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[])

        engine = RouterEngine(
            client=mock_backend,
            cache_enabled=False,
        )

        with patch("router.router.invalidate_all_caches") as mock_invalidate:
            engine.invalidate_caches()
            mock_invalidate.assert_called_once()
