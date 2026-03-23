"""Tests for provider_db module."""

import os
from pathlib import Path
import time
from unittest.mock import patch

import pytest

from router.provider_db import (
    ProviderDB,
    _provider_cache,
    get_provider_db,
    invalidate_provider_cache,
)


@pytest.fixture
def mock_provider_db_path(tmp_path):
    """Create a temporary provider.db file."""
    # We need to bypass the security check in ProviderDB.__init__
    import os

    os.environ["ROUTER_TEST_MODE"] = "true"
    try:
        db_path = tmp_path / "provider.db"
        yield str(db_path)
    finally:
        del os.environ["ROUTER_TEST_MODE"]


@pytest.fixture
def non_existent_path():
    import os

    os.environ["ROUTER_TEST_MODE"] = "true"
    try:
        yield "/nonexistent/path/provider.db"
    finally:
        del os.environ["ROUTER_TEST_MODE"]


@pytest.fixture
def sample_benchmark():
    """Sample benchmark data."""
    return {
        "model_id": "openai/gpt-4",
        "reasoning_score": 85.5,
        "coding_score": 90.0,
        "general_score": 88.0,
        "elo_rating": 1350,
    }


class TestProviderDB:
    """Tests for ProviderDB class."""

    def test_init_without_db(self, mock_provider_db_path):
        """Test initialization without database file."""
        db = ProviderDB(mock_provider_db_path)
        assert not db.is_available()

    def test_init_with_nonexistent_path(self):
        """Test initialization with nonexistent path."""
        with patch.dict(os.environ, {"ROUTER_TEST_MODE": "true"}):
            db = ProviderDB("/nonexistent/path/provider.db")
            assert not db.is_available()

    def test_get_stats_without_db(self, mock_provider_db_path):
        """Test get_stats when database doesn't exist."""
        db = ProviderDB(mock_provider_db_path)
        stats = db.get_stats()
        assert stats["available"] is False
        assert stats["total_models"] == 0

    def test_get_benchmark_without_db(self, mock_provider_db_path):
        """Test get_benchmark when database doesn't exist."""
        db = ProviderDB(mock_provider_db_path)
        result = db.get_benchmark("openai/gpt-4")
        assert result is None

    def test_get_benchmarks_for_models_empty(self, mock_provider_db_path):
        """Test get_benchmarks_for_models with empty list."""
        db = ProviderDB(mock_provider_db_path)
        result = db.get_benchmarks_for_models([])
        assert result == {}

    def test_resolve_alias_without_db(self, mock_provider_db_path):
        """Test resolve_alias when database doesn't exist."""
        db = ProviderDB(mock_provider_db_path)
        result = db.resolve_alias("gpt-4")
        assert result is None

    def test_find_model_by_name_without_db(self, mock_provider_db_path):
        """Test find_model_by_name when database doesn't exist."""
        db = ProviderDB(mock_provider_db_path)
        result = db.find_model_by_name("gpt-4")
        assert result is None


class TestProviderDBCache:
    """Tests for provider_db caching."""

    def test_cache_invalidation(self):
        """Test that cache can be invalidated."""
        # Import the module-level variables

        # Verify the function exists and runs without error
        invalidate_provider_cache()
        # If we got here without exception, test passes

    def test_fallback_returns_stale_cache_when_db_unavailable(self, mock_provider_db_path):
        """Fallback serves stale cache when provider.db is unavailable."""
        db = ProviderDB(mock_provider_db_path)

        # Prime cache with known benchmark map
        cached = {"openai/gpt-4": {"model_id": "openai/gpt-4", "general_score": 88.0}}
        _provider_cache.set("all_benchmarks", cached)

        import router.provider_db as provider_db_module

        provider_db_module._provider_db_last_good_cache_time = time.monotonic()

        with patch("router.provider_db.settings") as mock_settings:
            mock_settings.db_slow_fallback_enabled = True
            mock_settings.db_stale_cache_max_age_seconds = 300
            result = db.get_benchmarks_for_models(["openai/gpt-4"])

        assert "openai/gpt-4" in result

    def test_db_error_marks_degraded_and_uses_cache(self, mock_provider_db_path):
        """DB query errors trigger degraded mode and stale fallback."""
        db = ProviderDB(mock_provider_db_path)

        import router.provider_db as provider_db_module

        # Force availability so query path is reached
        with patch.object(db, "is_available", return_value=True):
            # Prime stale fallback cache
            cached = {"openai/gpt-4": {"model_id": "openai/gpt-4", "general_score": 88.0}}
            _provider_cache.set("all_benchmarks", cached)
            provider_db_module._provider_db_last_good_cache_time = time.monotonic()

            with (
                patch.object(db, "_get_connection", side_effect=RuntimeError("db down")),
                patch("router.provider_db.settings") as mock_settings,
            ):
                mock_settings.db_slow_fallback_enabled = True
                mock_settings.db_slow_fallback_window_seconds = 30
                mock_settings.db_stale_cache_max_age_seconds = 300
                # Include one uncached model to force DB query path (which errors)
                result = db.get_benchmarks_for_models(["openai/gpt-4", "anthropic/claude-3"])

            assert "openai/gpt-4" in result
            assert provider_db_module._provider_db_degraded_until > time.monotonic()

    def test_stats_reports_stale_when_last_build_old(self, mock_provider_db_path):
        """Stats include stale=true when metadata last_build exceeds threshold."""
        db = ProviderDB(mock_provider_db_path)

        class FakeCursor:
            def __init__(self):
                self.calls = 0

            def execute(self, *_args, **_kwargs):
                return None

            def fetchone(self):
                self.calls += 1
                if self.calls == 1:
                    return [10]  # total models
                if self.calls == 2:
                    return [1]  # archived
                return ["2000-01-01T00:00:00Z"]  # very old build

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def close(self):
                return None

        with (
            patch.object(db, "is_available", return_value=True),
            patch.object(db, "_get_connection", return_value=FakeConn()),
            patch.object(db, "_detect_archived_column", return_value=True),
            patch("router.provider_db.settings") as mock_settings,
        ):
            mock_settings.provider_db_max_age_hours = 1
            stats = db.get_stats()

        assert stats["available"] is True
        assert stats["stale"] is True
        assert "degraded" in stats


class TestProviderDBIntegration:
    """Integration tests for ProviderDB with actual database."""

    def test_get_provider_db_default_path(self):
        """Test get_provider_db with default path."""
        # This will check if provider.db exists at default location
        db = get_provider_db()
        # Just verify it returns a ProviderDB instance
        assert isinstance(db, ProviderDB)

    def test_real_provider_db_has_benchmarks(self):
        """Ensure real provider.db is readable and non-empty when present."""
        base_dir = Path(__file__).resolve().parents[1]
        db_path = base_dir / "data" / "provider.db"
        if not db_path.exists():
            pytest.skip("provider.db not present in repo")

        db = ProviderDB(str(db_path))
        assert db.is_available() is True
        stats = db.get_stats()
        assert stats["available"] is True
        assert stats["total_models"] > 0

        benchmarks = db.get_all_benchmarks()
        assert benchmarks
        assert "model_id" in benchmarks[0]

    @patch("router.provider_db.settings")
    def test_get_provider_db_with_custom_path(self, mock_settings, tmp_path):
        """Test get_provider_db with custom path."""
        custom_path = str(tmp_path / "custom.db")
        mock_settings.provider_db_path = custom_path

        with patch.dict(os.environ, {"ROUTER_TEST_MODE": "true"}):
            db = ProviderDB(custom_path)
            assert not db.is_available()  # Custom path doesn't exist


class TestProviderDBStats:
    """Tests for provider.db statistics."""

    @patch("router.provider_db.settings")
    def test_stats_with_nonexistent_db(self, mock_settings):
        """Test stats calculation with nonexistent db."""
        mock_settings.provider_db_path = "/nonexistent/provider.db"

        with patch.dict(os.environ, {"ROUTER_TEST_MODE": "true"}):
            db = ProviderDB("/nonexistent/provider.db")
            stats = db.get_stats()

            assert stats["available"] is False
            assert stats["total_models"] == 0


class TestProviderDBAliasResolution:
    """Tests for alias resolution."""

    def test_resolve_alias_returns_none_for_missing(self, mock_provider_db_path):
        """Test that resolve_alias returns None for missing aliases."""
        db = ProviderDB(mock_provider_db_path)
        result = db.resolve_alias("nonexistent-alias")
        assert result is None


class TestProviderDBModelLookup:
    """Tests for model lookup functionality."""

    def test_find_model_by_name_no_db(self, mock_provider_db_path):
        """Test find_model_by_name returns None when no DB."""
        db = ProviderDB(mock_provider_db_path)
        result = db.find_model_by_name("any-model")
        assert result is None

    def test_get_all_benchmarks_no_db(self, mock_provider_db_path):
        """Test get_all_benchmarks returns empty list when no DB."""
        db = ProviderDB(mock_provider_db_path)
        result = db.get_all_benchmarks()
        assert result == []
