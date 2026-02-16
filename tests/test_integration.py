"""Integration tests for the LLM Router system."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import after test client setup
from main import app, app_state


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_ollama_models():
    """Mock Ollama model list."""
    return [
        MagicMock(name="llama3", size=1000000000),
        MagicMock(name="codellama", size=2000000000),
    ]


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
    async def test_list_models(self, client):
        """Test /v1/models endpoint."""
        # Create proper mock objects with name attribute
        mock_model1 = MagicMock()
        mock_model1.name = "llama3"
        mock_model2 = MagicMock()
        mock_model2.name = "mistral"
        mock_models = [mock_model1, mock_model2]
        
        with patch.object(app_state, "backend") as mock_client:
            mock_client.list_models = AsyncMock(return_value=mock_models)
            
            response = client.get("/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 2
            assert data["data"][0]["id"] == "llama3"

    @pytest.mark.asyncio
    async def test_list_models_not_initialized(self, client):
        """Test model list when client not ready."""
        original_client = app_state.backend
        app_state.backend = None
        
        try:
            response = client.get("/v1/models")
            assert response.status_code == 503
        finally:
            app_state.backend = original_client


class TestChatEndpoints:
    """Test chat completion endpoints."""

    @pytest.mark.asyncio
    async def test_chat_completions_basic(self, client):
        """Test basic chat completion."""
        mock_response = {
            "message": {"content": "Hello! How can I help you?"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(app_state, "backend") as mock_client:
            with patch.object(app_state, "router_engine") as mock_router:
                mock_router.select_model = AsyncMock(return_value=MagicMock(
                    selected_model="llama3",
                    confidence=0.9,
                    reasoning="Matched to general profile"
                ))
                mock_client.chat = AsyncMock(return_value=mock_response)
                
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": "Hello!"}],
                    },
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["object"] == "chat.completion"
                assert "choices" in data
                assert "model" in data

    @pytest.mark.asyncio
    async def test_chat_completions_with_signature(self, client):
        """Test that model signature is appended."""
        mock_response = {
            "message": {"content": "Test response"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(app_state, "backend") as mock_client:
            with patch.object(app_state, "router_engine") as mock_router:
                mock_router.select_model = AsyncMock(return_value=MagicMock(
                    selected_model="llama3",
                    confidence=0.9,
                    reasoning="Test"
                ))
                mock_client.chat = AsyncMock(return_value=mock_response)
                
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Test"}],
                    },
                )
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                assert "llama3" in content  # Signature should be present

    @pytest.mark.asyncio
    async def test_chat_completions_no_messages(self, client):
        """Test error handling for missing messages."""
        # Initialize the app state for this test
        with patch.object(app_state, "backend", MagicMock()):
            with patch.object(app_state, "router_engine", MagicMock()):
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": []},
                )

                assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, client):
        """Test streaming chat completion."""
        async def mock_stream():
            chunks = [
                {"message": {"content": "Hello"}, "done": False},
                {"message": {"content": " world"}, "done": False},
                {"done": True},
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(app_state, "backend") as mock_client:
            with patch.object(app_state, "router_engine") as mock_router:
                mock_router.select_model = AsyncMock(return_value=MagicMock(
                    selected_model="llama3",
                    confidence=0.9,
                    reasoning="Test"
                ))
                mock_client.chat_streaming = AsyncMock(return_value=(mock_stream(), 0))
                
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Test"}],
                        "stream": True,
                    },
                )
                
                assert response.status_code == 200
                # Accept either exact match or with charset
                assert "text/event-stream" in response.headers["content-type"]


class TestAdminEndpoints:
    """Test admin endpoints."""

    @pytest.mark.asyncio
    async def test_get_profiles(self, client):
        """Test profiles admin endpoint."""
        # Mock the database session and query
        with patch("main.get_session") as mock_session:
            mock_context = MagicMock()
            mock_query = MagicMock()
            mock_query.all.return_value = []
            mock_context.query.return_value = mock_query
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_session.return_value.__exit__ = MagicMock(return_value=None)

            response = client.get("/admin/profiles")
            assert response.status_code == 200
            data = response.json()
            assert "profiles" in data

    @pytest.mark.asyncio
    async def test_get_benchmarks(self, client):
        """Test benchmarks admin endpoint."""
        with patch("main.get_session") as mock_session:
            with patch("main.get_last_sync") as mock_sync:
                mock_context = MagicMock()
                mock_query = MagicMock()
                mock_query.all.return_value = []
                mock_context.query.return_value = mock_query
                mock_session.return_value.__enter__ = MagicMock(return_value=mock_context)
                mock_session.return_value.__exit__ = MagicMock(return_value=None)
                mock_sync.return_value = None

                response = client.get("/admin/benchmarks")
                assert response.status_code == 200
                data = response.json()
                assert "benchmarks" in data

    @pytest.mark.asyncio
    async def test_reprofile(self, client):
        """Test manual reprofile endpoint."""
        from router.profiler import ProfileResult
        
        # Create a mock ollama client that won't fail
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(return_value=[])
        
        with patch.object(app_state, "backend", mock_client):
            with patch("main.profile_all_models", new_callable=AsyncMock) as mock_profile:
                mock_profile.return_value = [
                    ProfileResult("llama3", 0.8, 0.7, 0.6, 0.9, 0.75, 1200.0)
                ]

                response = client.post("/admin/reprofile")
                assert response.status_code == 200
                data = response.json()
                assert "profiled" in data


class TestRouterIntegration:
    """Test the full routing flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_routing(self, client):
        """Test complete request flow from start to finish."""
        # This would test:
        # 1. User sends request
        # 2. Router selects model based on prompt
        # 3. Model generates response
        # 4. Response returned with signature
        
        mock_response = {
            "message": {"content": "I am a helpful assistant."},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }

        mock_backend = MagicMock()
        mock_backend.chat = AsyncMock(return_value=mock_response)
        mock_backend.unload_model = AsyncMock(return_value=True)
        
        with patch.object(app_state, "backend", mock_backend):
            with patch.object(app_state, "router_engine") as mock_router:
                # Simulate routing to different models based on prompt
                mock_router.select_model = AsyncMock(return_value=MagicMock(
                    selected_model="codellama",
                    confidence=0.95,
                    reasoning="Coding task detected"
                ))
                
                # Test coding prompt
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Write a Python function"}],
                    },
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify router was called
                mock_router.select_model.assert_called_once()
                
                # Verify model was called
                mock_backend.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_routing_fallback(self, client):
        """Test fallback when routing fails."""
        mock_models = [MagicMock(name="llama3")]

        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=mock_models)
        mock_backend.chat = AsyncMock(return_value={
            "message": {"content": "Fallback response"},
            "prompt_eval_count": 5,
            "eval_count": 5,
        })
        mock_backend.unload_model = AsyncMock(return_value=True)
        
        with patch.object(app_state, "backend", mock_backend):
            with patch.object(app_state, "router_engine") as mock_router:
                mock_router.select_model = AsyncMock(side_effect=Exception("Router failed"))
                
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "Hello"}]},
                )
                
                # Should still succeed with fallback
                assert response.status_code == 200
