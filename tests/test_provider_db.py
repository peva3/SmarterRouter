"""Tests for provider_db module."""

import os
from unittest.mock import patch

import pytest

from router.provider_db import (
    ProviderDB,
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


class TestProviderDBIntegration:
    """Integration tests for ProviderDB with actual database."""

    def test_get_provider_db_default_path(self):
        """Test get_provider_db with default path."""
        # This will check if provider.db exists at default location
        db = get_provider_db()
        # Just verify it returns a ProviderDB instance
        assert isinstance(db, ProviderDB)

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
