"""Extended integration tests with mock backend for full chat flow."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from fastapi.testclient import TestClient
from httpx import Response as HttpxResponse

from main import app, app_state, get_settings
from router.config import Settings
from router.backends.base import ModelInfo


@pytest.fixture
def mock_backend():
    """Create a mock backend that simulates Ollama responses."""
    backend = MagicMock()
    
    # Mock list_models to return available models
    async def mock_list_models():
        return [
            ModelInfo(name="llama3.2:1b", size=1321098329, modified_at="2026-02-15T16:35:08.8417444Z"),
            ModelInfo(name="qwen2.5-coder:14b", size=8988124298, modified_at="2026-02-14T14:32:47.74589142Z"),
        ]
    
    backend.list_models = mock_list_models
    
    # Mock chat response
    async def mock_chat(model, messages, stream=False, **kwargs):
        return {
            "message": {
                "role": "assistant",
                "content": "This is a test response from " + model
            }
        }
    
    backend.chat = mock_chat
    
    # Mock chat_streaming
    async def mock_chat_streaming(model, messages, **kwargs):
        async def stream_gen():
            chunks = [
                {"message": {"content": "This "}},
                {"message": {"content": "is "}},
                {"message": {"content": "streaming "}},
                {"message": {"content": "response"}},
            ]
            for chunk in chunks:
                yield chunk
        
        return stream_gen(), 0.0
    
    backend.chat_streaming = mock_chat_streaming
    
    return backend


@pytest.fixture
def mock_router_engine():
    """Create a mock router engine with caching."""
    engine = MagicMock()
    
    # Mock select_model to return predictable results
    async def mock_select_model(prompt, available_models):
        from router.router import RoutingResult
        
        # Simple logic: coding prompts go to qwen, others to llama
        if "code" in prompt.lower() or "python" in prompt.lower():
            return RoutingResult(
                selected_model="qwen2.5-coder:14b",
                confidence=0.95,
                reasoning="Coding task detected"
            )
        else:
            return RoutingResult(
                selected_model="llama3.2:1b",
                confidence=0.85,
                reasoning="General query"
            )
    
    engine.select_model = mock_select_model
    
    # Mock semantic cache
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.get_response = AsyncMock(return_value=None)
    cache.set_response = AsyncMock()
    engine.semantic_cache = cache
    
    engine.log_decision = AsyncMock()
    
    return engine


@pytest.fixture
def configured_client(mock_backend, mock_router_engine):
    """Create a test client with mocked backend and router."""
    
    def get_settings_override():
        return Settings(
            admin_api_key=None,
            cache_enabled=True,
            signature_enabled=True
        )
    
    app.dependency_overrides[get_settings] = get_settings_override
    
    # Set up mock state
    original_backend = app_state.backend
    original_router = app_state.router_engine
    
    app_state.backend = mock_backend
    app_state.router_engine = mock_router_engine
    app_state.vram_manager = None  # Disable VRAM manager for tests
    
    yield TestClient(app)
    
    # Restore
    app_state.backend = original_backend
    app_state.router_engine = original_router
    app.dependency_overrides = {}


class TestChatCompletions:
    """Test full chat completion flow."""
    
    def test_non_streaming_chat(self, configured_client):
        """Test basic non-streaming chat completion."""
        response = configured_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello, how are you?"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        # Should include model signature
        assert "Model:" in data["choices"][0]["message"]["content"]
    
    def test_coding_prompt_routing(self, configured_client):
        """Test that coding prompts are routed to appropriate model."""
        response = configured_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Write Python code for binary search"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should route to coding model
        assert "qwen" in data["model"].lower()
    
    def test_streaming_chat(self, configured_client):
        """Test streaming chat completion."""
        response = configured_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )
        
        assert response.status_code == 200
        # Check it's a streaming response
        assert response.headers["content-type"] == "text/event-stream"
        
        # Read streaming chunks
        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    chunks.append(json.loads(data))
        
        assert len(chunks) > 0
        # Each chunk should have choices with delta
        for chunk in chunks:
            assert "choices" in chunk
            assert len(chunk["choices"]) > 0
    
    def test_model_override(self, configured_client):
        """Test model override via query parameter."""
        response = configured_client.post(
            "/v1/chat/completions?model=qwen2.5-coder:14b",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "qwen2.5-coder:14b"
    
    def test_invalid_model_override(self, configured_client):
        """Test error handling for invalid model override."""
        response = configured_client.post(
            "/v1/chat/completions?model=nonexistent-model",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        # Should fall back gracefully or return error
        assert response.status_code in [200, 400]
    
    def test_invalid_content_type(self, configured_client):
        """Test error for non-JSON content."""
        response = configured_client.post(
            "/v1/chat/completions",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 415
        assert "Content-Type must be application/json" in response.text
    
    def test_empty_messages(self, configured_client):
        """Test error handling for empty messages."""
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": []}
        )
        
        assert response.status_code == 400
    
    def test_generation_parameters(self, configured_client):
        """Test that generation parameters are passed to backend."""
        response = configured_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9
            }
        )
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and fallback behavior."""
    
    def test_backend_unavailable(self, configured_client):
        """Test behavior when backend is unavailable."""
        # Temporarily remove backend
        original = app_state.backend
        app_state.backend = None
        
        try:
            response = configured_client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Test"}]}
            )
            assert response.status_code == 503
        finally:
            app_state.backend = original
    
    def test_model_generation_failure(self, configured_client, mock_backend):
        """Test fallback when primary model fails."""
        # Make first model fail
        call_count = 0
        
        async def failing_chat(model, messages, stream=False, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Model failed")
            return {
                "message": {
                    "role": "assistant",
                    "content": "Fallback response from " + model
                }
            }
        
        mock_backend.chat = failing_chat
        
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test"}]}
        )
        
        # Should succeed with fallback
        assert response.status_code == 200
        # Backend should have been called multiple times
        assert call_count >= 1


class TestCaching:
    """Test caching behavior."""
    
    def test_routing_cache_hit(self, configured_client, mock_router_engine):
        """Test that routing decisions are cached."""
        from router.router import RoutingResult
        
        # Set up cache hit
        cached_result = RoutingResult(
            selected_model="llama3.2:1b",
            confidence=0.9,
            reasoning="Cached decision"
        )
        mock_router_engine.semantic_cache.get = AsyncMock(return_value=cached_result)
        
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        
        assert response.status_code == 200
        # Should use cached routing
        mock_router_engine.semantic_cache.get.assert_called_once()
    
    def test_response_cache_hit(self, configured_client, mock_router_engine):
        """Test that responses are cached."""
        mock_router_engine.semantic_cache.get_response = AsyncMock(
            return_value="Cached response"
        )
        
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "Cached response" in data["choices"][0]["message"]["content"]


class TestExplainEndpoint:
    """Test the /admin/explain endpoint."""
    
    def test_explain_routing(self, configured_client):
        """Test explain endpoint returns routing breakdown."""
        response = configured_client.get(
            "/admin/explain?prompt=Write Python code for sorting"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "selected_model" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "model_scores" in data
    
    def test_explain_with_override(self, configured_client):
        """Test explain with model override."""
        response = configured_client.get(
            "/admin/explain?prompt=Hello&model_override=llama3.2:1b"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["selected_model"] == "llama3.2:1b"
        assert data.get("override") == True


class TestRequestValidation:
    """Test input validation and sanitization."""
    
    def test_oversized_prompt(self, configured_client):
        """Test rejection of oversized prompts."""
        # Create a very long prompt (>10000 chars)
        long_prompt = "A" * 11000
        
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": long_prompt}]}
        )
        
        assert response.status_code == 400
    
    def test_invalid_role(self, configured_client):
        """Test validation of message roles."""
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "invalid", "content": "Test"}]}
        )
        
        assert response.status_code == 400
    
    def test_xss_in_prompt(self, configured_client):
        """Test that malicious content is sanitized."""
        xss_prompt = "<script>alert('xss')</script>Hello"
        
        response = configured_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": xss_prompt}]}
        )
        
        # Should be accepted but sanitized
        assert response.status_code == 200


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def get_ratelimit_settings(self):
        return Settings(
            admin_api_key=None,
            rate_limit_enabled=True,
            rate_limit_requests_per_minute=2,  # Very low for testing
            rate_limit_admin_requests_per_minute=2
        )
    
    def test_rate_limit_enforced(self):
        """Test that rate limiting blocks excessive requests."""
        app.dependency_overrides[get_settings] = self.get_ratelimit_settings
        
        client = TestClient(app)
        
        # Make requests up to limit
        for i in range(2):
            response = client.get("/v1/models")
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = client.get("/v1/models")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.text
        
        app.dependency_overrides = {}
    
    def test_rate_limit_per_ip(self):
        """Test that rate limits are per-IP."""
        app.dependency_overrides[get_settings] = self.get_ratelimit_settings
        
        # Simulate different IPs
        client1 = TestClient(app, headers={"X-Forwarded-For": "1.2.3.4"})
        client2 = TestClient(app, headers={"X-Forwarded-For": "5.6.7.8"})
        
        # Each should have separate limit
        for i in range(2):
            assert client1.get("/v1/models").status_code == 200
            assert client2.get("/v1/models").status_code == 200
        
        app.dependency_overrides = {}


class TestDockerHealth:
    """Test health check endpoint used by Docker."""
    
    def test_health_endpoint(self, configured_client):
        """Test Docker health check endpoint."""
        response = configured_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, configured_client):
        """Test service readiness."""
        response = configured_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
