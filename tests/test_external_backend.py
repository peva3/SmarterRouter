"""Tests for External Backend Factory module."""

from unittest.mock import patch

from router.backends.external import (
    PROVIDER_CONFIGS,
    ExternalBackendFactory,
    get_external_backend_factory,
)


class TestExternalBackendFactory:
    """Tests for ExternalBackendFactory class."""

    def test_get_provider_from_model_with_prefix(self):
        """Test extracting provider from model name with prefix."""
        factory = ExternalBackendFactory()

        assert factory._get_provider_from_model("openai/gpt-4") == "openai"
        assert factory._get_provider_from_model("anthropic/claude-3-opus") == "anthropic"
        assert factory._get_provider_from_model("google/gemini-1.5-pro") == "google"
        assert factory._get_provider_from_model("cohere/command-r-plus") == "cohere"
        assert factory._get_provider_from_model("mistral/mistral-large") == "mistral"

    def test_get_provider_from_model_without_prefix(self):
        """Test that model names without prefix return None."""
        factory = ExternalBackendFactory()

        assert factory._get_provider_from_model("gpt-4") is None
        assert factory._get_provider_from_model("llama3") is None

    def test_is_external_model(self):
        """Test is_external_model detection."""
        factory = ExternalBackendFactory()

        assert factory.is_external_model("openai/gpt-4") is True
        assert factory.is_external_model("anthropic/claude-3") is True
        assert factory.is_external_model("gpt-4") is False
        assert factory.is_external_model("llama3") is False

    @patch("router.backends.external.settings")
    def test_get_backend_returns_none_without_api_key(self, mock_settings):
        """Test that backend returns None when no API key configured."""
        mock_settings.external_providers = ["openai", "anthropic", "google"]
        mock_settings.openai_api_key = None
        mock_settings.generation_timeout = 120

        factory = ExternalBackendFactory()
        backend = factory.get_backend("openai/gpt-4")

        assert backend is None

    @patch("router.backends.external.settings")
    def test_get_backend_creates_backend_with_api_key(self, mock_settings):
        """Test that backend is created when API key is configured."""
        mock_settings.external_providers = ["openai", "anthropic", "google"]
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_base_url = None
        mock_settings.generation_timeout = 120

        factory = ExternalBackendFactory()
        backend = factory.get_backend("openai/gpt-4")

        assert backend is not None
        assert backend.api_key == "test-key"

    @patch("router.backends.external.settings")
    def test_get_backend_caches_created_backends(self, mock_settings):
        """Test that backends are cached after creation."""
        mock_settings.external_providers = ["openai", "anthropic", "google"]
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_base_url = None
        mock_settings.generation_timeout = 120

        factory = ExternalBackendFactory()

        backend1 = factory.get_backend("openai/gpt-4")
        backend2 = factory.get_backend("openai/gpt-4o")

        assert backend1 is backend2  # Same cached instance

    @patch("router.backends.external.settings")
    def test_get_backend_returns_none_for_disabled_provider(self, mock_settings):
        """Test that backend returns None for disabled provider."""
        mock_settings.external_providers = ["anthropic"]  # OpenAI not enabled
        mock_settings.openai_api_key = "test-key"
        mock_settings.generation_timeout = 120

        factory = ExternalBackendFactory()
        backend = factory.get_backend("openai/gpt-4")

        assert backend is None

    @patch("router.backends.external.settings")
    def test_get_backend_returns_none_for_unknown_provider(self, mock_settings):
        """Test that backend returns None for unknown provider."""
        mock_settings.external_providers = ["openai", "anthropic", "google"]

        factory = ExternalBackendFactory()
        backend = factory.get_backend("unknown/model")

        assert backend is None


class TestProviderConfigs:
    """Tests for provider configurations."""

    def test_all_providers_have_required_fields(self):
        """Test that all providers have required configuration fields."""
        required_fields = ["default_base_url", "api_key_field"]

        for provider, config in PROVIDER_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Provider {provider} missing {field}"

    def test_anthropic_uses_anthropic_base_url(self):
        """Test Anthropic configuration."""
        config = PROVIDER_CONFIGS["anthropic"]
        assert "anthropic.com" in config["default_base_url"]
        assert config["api_key_field"] == "anthropic_api_key"

    def test_google_uses_google_base_url(self):
        """Test Google configuration."""
        config = PROVIDER_CONFIGS["google"]
        assert "generativelanguage.googleapis.com" in config["default_base_url"]
        assert config["api_key_field"] == "google_api_key"


class TestGetExternalBackendFactory:
    """Tests for get_external_backend_factory function."""

    def test_returns_singleton(self):
        """Test that function returns singleton instance."""
        # Reset global
        import router.backends.external as extmod

        extmod._external_backend_factory = None

        factory1 = get_external_backend_factory()
        factory2 = get_external_backend_factory()

        assert factory1 is factory2
