"""Integration tests for the SmarterRouter system."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import Depends

from main import app, app_state, get_settings
from router.config import Settings


@pytest.fixture
def client():
    """Create a test client that can be configured for auth."""

    def get_settings_override_unauthed():
        return Settings(admin_api_key=None)

    app.dependency_overrides[get_settings] = get_settings_override_unauthed
    yield TestClient(app)
    app.dependency_overrides = {}


@pytest.fixture
def authed_client():
    """Create an authenticated test client."""

    def get_settings_override_authed():
        return Settings(admin_api_key="test-key")

    app.dependency_overrides[get_settings] = get_settings_override_authed
    yield TestClient(app, headers={"Authorization": "Bearer test-key"})
    app.dependency_overrides = {}


class TestHealthEndpoints:
    """Test basic health and status endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestModelEndpoints:
    """Test model listing and selection endpoints."""

    @pytest.mark.asyncio
    async def test_list_models_auth_not_required(self, client):
        """Test /v1/models when no admin key is set."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["admin_auth_required"] == False

    @pytest.mark.asyncio
    async def test_list_models_auth_required(self, authed_client):
        """Test /v1/models when admin key is set."""
        response = authed_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["admin_auth_required"] == True

    @pytest.mark.asyncio
    async def test_list_models_not_initialized(self, client):
        """Test model list when client not ready (should still work)."""
        original_backend = app_state.backend
        app_state.backend = None
        try:
            response = client.get("/v1/models")
            assert response.status_code == 200  # Endpoint is independent of backend state
        finally:
            app_state.backend = original_backend


class TestAdminEndpoints:
    """Test admin endpoints with and without authentication."""

    def test_get_profiles_unauthorized(self, client):
        """Test admin endpoint without auth when it's required (should fail)."""

        # We need to simulate a state where auth is required for this client
        def get_settings_override():
            return Settings(admin_api_key="a-key-is-set")

        app.dependency_overrides[get_settings] = get_settings_override
        response = client.get("/admin/profiles")
        assert response.status_code == 401
        app.dependency_overrides = {}

    def test_get_profiles_authorized(self, authed_client):
        """Test admin endpoint with correct authentication."""
        with patch("main.get_session"):
            response = authed_client.get("/admin/profiles")
            assert response.status_code == 200

    def test_get_benchmarks_unauthorized(self, client):
        def get_settings_override():
            return Settings(admin_api_key="a-key-is-set")

        app.dependency_overrides[get_settings] = get_settings_override
        response = client.get("/admin/benchmarks")
        assert response.status_code == 401
        app.dependency_overrides = {}

    def test_get_benchmarks_authorized(self, authed_client):
        with patch("main.get_session"):
            with patch("main.get_last_sync"):
                response = authed_client.get("/admin/benchmarks")
                assert response.status_code == 200

    def test_reprofile_unauthorized(self, client):
        def get_settings_override():
            return Settings(admin_api_key="a-key-is-set")

        app.dependency_overrides[get_settings] = get_settings_override
        response = client.post("/admin/reprofile")
        assert response.status_code == 401
        app.dependency_overrides = {}

    @patch("main.profile_all_models", new_callable=AsyncMock)
    def test_reprofile_authorized(self, mock_profile, authed_client):
        mock_profile.return_value = []
        app_state.backend = MagicMock()
        response = authed_client.post("/admin/reprofile")
        assert response.status_code == 200
