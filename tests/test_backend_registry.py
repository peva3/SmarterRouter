"""Tests for BackendRegistry module."""

from unittest.mock import AsyncMock, MagicMock, patch

from router.backends.base import ModelInfo
from router.backends.registry import BackendRegistry, get_backend_registry


class TestBackendRegistry:
    """Tests for BackendRegistry class."""

    @patch("router.backends.registry.create_backend")
    @patch("router.backends.registry.get_provider_db")
    def test_init_without_local_backend(self, mock_provider_db, mock_create_backend):
        """Test initialization without local backend."""
        mock_create_backend.side_effect = Exception("No backend")

        with patch("router.backends.registry.settings") as mock_settings:
            mock_settings.provider = "ollama"
            mock_settings.provider_db_enabled = False
            mock_settings.external_providers_enabled = False

            registry = BackendRegistry()
            registry.initialize()

            assert registry.local_backend is None

    @patch("router.backends.registry.create_backend")
    @patch("router.backends.registry.get_provider_db")
    def test_init_with_local_backend(self, mock_provider_db, mock_create_backend):
        """Test initialization with local backend."""
        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[])
        mock_create_backend.return_value = mock_backend

        mock_provider_db_instance = MagicMock()
        mock_provider_db_instance.is_available.return_value = False
        mock_provider_db.return_value = mock_provider_db_instance

        with patch("router.backends.registry.settings") as mock_settings:
            mock_settings.provider = "ollama"
            mock_settings.provider_db_enabled = False
            mock_settings.external_providers_enabled = False
            mock_settings.model_filter_include = []

            registry = BackendRegistry()
            registry.initialize()

            assert registry.local_backend is not None

    def test_is_local_model_without_backend(self):
        """Test is_local_model returns False without backend."""
        registry = BackendRegistry()
        assert registry.is_local_model("test-model") is False

    def test_is_external_model_without_db(self):
        """Test is_external_model returns False without provider.db."""
        registry = BackendRegistry()
        assert registry.is_external_model("openai/gpt-4") is False

    @patch("router.backends.registry.create_backend")
    @patch("router.backends.registry.get_provider_db")
    def test_get_backend_for_model_unknown(self, mock_provider_db, mock_create_backend):
        """Test get_backend_for_model returns unknown for unrecognized models."""
        mock_backend = MagicMock()
        mock_create_backend.return_value = mock_backend

        mock_provider_db_instance = MagicMock()
        mock_provider_db_instance.is_available.return_value = False
        mock_provider_db_instance.get_benchmark.return_value = None
        mock_provider_db.return_value = mock_provider_db_instance

        with patch("router.backends.registry.settings") as mock_settings:
            mock_settings.provider = "ollama"
            mock_settings.provider_db_enabled = True
            mock_settings.external_providers_enabled = True
            mock_settings.model_filter_include = []

            registry = BackendRegistry()
            registry.initialize()

            backend_type, backend = registry.get_backend_for_model("unknown-model")
            # When local backend exists, defaults to local
            assert backend_type in ("local", "external")

    def test_close_without_backend(self):
        """Test close works without backend."""
        registry = BackendRegistry()
        # Should not raise
        import asyncio

        asyncio.run(registry.close())


class TestBackendRegistryProviderDB:
    """Tests for BackendRegistry with provider.db."""

    @patch("router.backends.registry.create_backend")
    @patch("router.backends.registry.get_provider_db")
    def test_is_external_model_with_db(self, mock_provider_db, mock_create_backend):
        """Test is_external_model returns True when model in provider.db."""
        mock_backend = MagicMock()
        mock_create_backend.return_value = mock_backend

        mock_provider_db_instance = MagicMock()
        mock_provider_db_instance.is_available.return_value = True
        mock_provider_db_instance.get_benchmark.return_value = {
            "model_id": "openai/gpt-4",
            "reasoning_score": 85.0,
        }
        mock_provider_db.return_value = mock_provider_db_instance

        with patch("router.backends.registry.settings") as mock_settings:
            mock_settings.provider = "ollama"
            mock_settings.provider_db_enabled = True
            mock_settings.external_providers_enabled = True
            mock_settings.model_filter_include = []

            registry = BackendRegistry()
            registry.initialize()

            assert registry.is_external_model("openai/gpt-4") is True

    @patch("router.backends.registry.create_backend")
    @patch("router.backends.registry.get_provider_db")
    def test_list_models_includes_external(self, mock_provider_db, mock_create_backend):
        """Test list_models includes external models when enabled."""
        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[ModelInfo(name="llama3", size=3.8e9)])
        mock_create_backend.return_value = mock_backend

        mock_provider_db_instance = MagicMock()
        mock_provider_db_instance.is_available.return_value = True
        mock_provider_db_instance.get_all_benchmarks.return_value = [
            {"model_id": "openai/gpt-4", "reasoning_score": 85.0},
            {"model_id": "anthropic/claude-3", "reasoning_score": 90.0},
        ]
        mock_provider_db.return_value = mock_provider_db_instance

        with patch("router.backends.registry.settings") as mock_settings:
            mock_settings.provider = "ollama"
            mock_settings.provider_db_enabled = True
            mock_settings.external_providers_enabled = True
            mock_settings.model_filter_include = []

            registry = BackendRegistry()
            registry.initialize()

            import asyncio

            models = asyncio.run(registry.list_models())

            # Should have local + external models
            model_names = [m.name for m in models]
            assert "llama3" in model_names
            assert "openai/gpt-4" in model_names
            assert "anthropic/claude-3" in model_names


class TestGetBackendRegistry:
    """Tests for get_backend_registry function."""

    def test_get_backend_registry_singleton(self):
        """Test that get_backend_registry returns singleton."""
        # Reset global
        import router.backends.registry as reg

        reg._backend_registry = None

        # Should return same instance
        with patch.object(BackendRegistry, "initialize"):
            reg1 = get_backend_registry()
            reg2 = get_backend_registry()
            assert reg1 is reg2
