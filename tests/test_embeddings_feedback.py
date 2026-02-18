"""Tests for Embeddings and Feedback endpoints."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from main import app, app_state, get_settings
from router.config import Settings
from router.backends.base import ModelInfo


class TestEmbeddingsEndpoint:
    """Test /v1/embeddings endpoint."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend with embeddings support."""
        backend = MagicMock()
        
        async def mock_embed(model, input_text, **kwargs):
            if isinstance(input_text, list):
                return {
                    "embeddings": [[0.1, 0.2, 0.3] for _ in input_text],
                    "usage": {"prompt_tokens": len(input_text) * 10, "total_tokens": len(input_text) * 10}
                }
            return {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "usage": {"prompt_tokens": 10, "total_tokens": 10}
            }
        
        backend.embed = mock_embed
        return backend

    @pytest.fixture
    def client_with_backend(self, mock_backend):
        """Create a test client with mocked backend."""
        def get_settings_override():
            return Settings(admin_api_key=None)
        
        app.dependency_overrides[get_settings] = get_settings_override
        
        original_backend = app_state.backend
        app_state.backend = mock_backend
        
        yield TestClient(app)
        
        app_state.backend = original_backend
        app.dependency_overrides = {}

    def test_embeddings_single_text(self, client_with_backend):
        """Test embeddings with single text input."""
        response = client_with_backend.post(
            "/v1/embeddings",
            json={
                "model": "nomic-embed-text",
                "input": "Hello, world!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]

    def test_embeddings_multiple_texts(self, client_with_backend):
        """Test embeddings with multiple text inputs."""
        response = client_with_backend.post(
            "/v1/embeddings",
            json={
                "model": "nomic-embed-text",
                "input": ["Hello", "World"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

    def test_embeddings_with_encoding_format(self, client_with_backend):
        """Test embeddings with encoding format parameter."""
        response = client_with_backend.post(
            "/v1/embeddings",
            json={
                "model": "nomic-embed-text",
                "input": "Test text",
                "encoding_format": "float"
            }
        )
        
        assert response.status_code == 200

    def test_embeddings_missing_model(self, client_with_backend):
        """Test embeddings without model parameter."""
        response = client_with_backend.post(
            "/v1/embeddings",
            json={"input": "Test"}
        )
        
        assert response.status_code == 400

    def test_embeddings_missing_input(self, client_with_backend):
        """Test embeddings without input parameter."""
        response = client_with_backend.post(
            "/v1/embeddings",
            json={"model": "nomic-embed-text"}
        )
        
        assert response.status_code == 400

    def test_embeddings_invalid_content_type(self, client_with_backend):
        """Test embeddings with invalid content type."""
        response = client_with_backend.post(
            "/v1/embeddings",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 415

    def test_embeddings_backend_unavailable(self):
        """Test embeddings when backend is not available."""
        def get_settings_override():
            return Settings(admin_api_key=None)
        
        app.dependency_overrides[get_settings] = get_settings_override
        
        original_backend = app_state.backend
        app_state.backend = None
        
        try:
            client = TestClient(app)
            response = client.post(
                "/v1/embeddings",
                json={"model": "test", "input": "test"}
            )
            assert response.status_code == 503
        finally:
            app_state.backend = original_backend
            app.dependency_overrides = {}

    def test_embeddings_openai_format_response(self, mock_backend):
        """Test handling OpenAI format embedding response."""
        async def mock_embed_openai(model, input_text, **kwargs):
            return {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3], "index": 0}
                ],
                "usage": {"prompt_tokens": 10, "total_tokens": 10}
            }
        
        mock_backend.embed = mock_embed_openai
        
        def get_settings_override():
            return Settings(admin_api_key=None)
        
        app.dependency_overrides[get_settings] = get_settings_override
        original_backend = app_state.backend
        app_state.backend = mock_backend
        
        try:
            client = TestClient(app)
            response = client.post(
                "/v1/embeddings",
                json={"model": "test", "input": "test"}
            )
            assert response.status_code == 200
        finally:
            app_state.backend = original_backend
            app.dependency_overrides = {}


class TestFeedbackEndpoint:
    """Test /v1/feedback endpoint."""

    @pytest.fixture
    def client_with_feedback(self):
        """Create a test client with feedback enabled."""
        def get_settings_override():
            return Settings(
                admin_api_key=None,
                feedback_enabled=True
            )
        
        app.dependency_overrides[get_settings] = get_settings_override
        
        yield TestClient(app)
        
        app.dependency_overrides = {}

    @pytest.fixture
    def client_feedback_disabled(self):
        """Create a test client with feedback disabled."""
        def get_settings_override():
            return Settings(
                admin_api_key=None,
                feedback_enabled=False
            )
        
        app.dependency_overrides[get_settings] = get_settings_override
        
        yield TestClient(app)
        
        app.dependency_overrides = {}

    def test_feedback_positive(self, client_with_feedback):
        """Test submitting positive feedback."""
        with patch("main.get_session") as mock_session:
            mock_session.return_value.__enter__.return_value.add = MagicMock()
            mock_session.return_value.__enter__.return_value.commit = MagicMock()
            
            response = client_with_feedback.post(
                "/v1/feedback",
                json={
                    "model_name": "llama3",
                    "score": 1.0,
                    "comment": "Great response!"
                }
            )
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"

    def test_feedback_negative(self, client_with_feedback):
        """Test submitting negative feedback."""
        with patch("main.get_session") as mock_session:
            mock_session.return_value.__enter__.return_value.add = MagicMock()
            mock_session.return_value.__enter__.return_value.commit = MagicMock()
            
            response = client_with_feedback.post(
                "/v1/feedback",
                json={
                    "model_name": "llama3",
                    "score": -1.0,
                    "comment": "Poor response"
                }
            )
            
            assert response.status_code == 200

    def test_feedback_with_category(self, client_with_feedback):
        """Test submitting feedback with category."""
        with patch("main.get_session") as mock_session:
            mock_session.return_value.__enter__.return_value.add = MagicMock()
            mock_session.return_value.__enter__.return_value.commit = MagicMock()
            
            response = client_with_feedback.post(
                "/v1/feedback",
                json={
                    "model_name": "codellama",
                    "score": 0.8,
                    "category": "coding",
                    "comment": "Good code generation"
                }
            )
            
            assert response.status_code == 200

    def test_feedback_with_response_id(self, client_with_feedback):
        """Test submitting feedback with response ID linking."""
        with patch("main.get_session") as mock_session:
            mock_session_obj = mock_session.return_value.__enter__.return_value
            mock_session_obj.add = MagicMock()
            mock_session_obj.commit = MagicMock()
            mock_session_obj.execute.return_value.scalar_one_or_none.return_value = None
            
            response = client_with_feedback.post(
                "/v1/feedback",
                json={
                    "response_id": "chatcmpl-abc123",
                    "score": 1.0
                }
            )
            
            assert response.status_code == 400

    def test_feedback_missing_model_name_no_response_id(self, client_with_feedback):
        """Test feedback fails without model_name or valid response_id."""
        with patch("main.get_session") as mock_session:
            mock_session_obj = mock_session.return_value.__enter__.return_value
            mock_session_obj.execute.return_value.scalar_one_or_none.return_value = None
            
            response = client_with_feedback.post(
                "/v1/feedback",
                json={"score": 1.0}
            )
            
            assert response.status_code == 400

    def test_feedback_disabled(self, client_feedback_disabled):
        """Test feedback returns 403 when disabled."""
        response = client_feedback_disabled.post(
            "/v1/feedback",
            json={
                "model_name": "llama3",
                "score": 1.0
            }
        )
        
        assert response.status_code == 403

    def test_feedback_score_bounds_high(self, client_with_feedback):
        """Test feedback with score above 1.0 is rejected."""
        response = client_with_feedback.post(
            "/v1/feedback",
            json={
                "model_name": "llama3",
                "score": 2.0
            }
        )
        
        assert response.status_code == 422

    def test_feedback_score_bounds_low(self, client_with_feedback):
        """Test feedback with score below -1.0 is rejected."""
        response = client_with_feedback.post(
            "/v1/feedback",
            json={
                "model_name": "llama3",
                "score": -2.0
            }
        )
        
        assert response.status_code == 422

    def test_feedback_long_comment(self, client_with_feedback):
        """Test feedback with comment exceeding max length."""
        long_comment = "x" * 600
        
        response = client_with_feedback.post(
            "/v1/feedback",
            json={
                "model_name": "llama3",
                "score": 1.0,
                "comment": long_comment
            }
        )
        
        assert response.status_code == 422

    def test_feedback_database_error(self, client_with_feedback):
        """Test feedback handles database errors gracefully."""
        with patch("main.get_session") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB error")
            
            response = client_with_feedback.post(
                "/v1/feedback",
                json={
                    "model_name": "llama3",
                    "score": 1.0
                }
            )
            
            assert response.status_code == 500


class TestFeedbackRequestValidation:
    """Test FeedbackRequest schema validation."""

    def test_valid_feedback_request(self):
        """Test valid feedback request schema."""
        from router.schemas import FeedbackRequest
        
        feedback = FeedbackRequest(
            model_name="llama3",
            score=1.0,
            comment="Great!",
            category="coding"
        )
        
        assert feedback.model_name == "llama3"
        assert feedback.score == 1.0

    def test_feedback_request_minimal(self):
        """Test minimal feedback request."""
        from router.schemas import FeedbackRequest
        
        feedback = FeedbackRequest(score=0.5)
        
        assert feedback.model_name is None
        assert feedback.response_id is None

    def test_feedback_request_score_validation(self):
        """Test score boundary validation."""
        from router.schemas import FeedbackRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            FeedbackRequest(score=1.5)
        
        with pytest.raises(ValidationError):
            FeedbackRequest(score=-1.5)


class TestEmbeddingsRequestValidation:
    """Test EmbeddingsRequest schema validation."""

    def test_valid_embeddings_request(self):
        """Test valid embeddings request schema."""
        from router.schemas import EmbeddingsRequest
        
        req = EmbeddingsRequest(
            model="nomic-embed-text",
            input="Hello world"
        )
        
        assert req.model == "nomic-embed-text"
        assert req.input == "Hello world"

    def test_embeddings_request_list_input(self):
        """Test embeddings request with list input."""
        from router.schemas import EmbeddingsRequest
        
        req = EmbeddingsRequest(
            model="nomic-embed-text",
            input=["Hello", "World"]
        )
        
        assert isinstance(req.input, list)
        assert len(req.input) == 2

    def test_embeddings_request_encoding_format(self):
        """Test encoding format parameter."""
        from router.schemas import EmbeddingsRequest
        
        req = EmbeddingsRequest(
            model="test",
            input="test",
            encoding_format="base64"
        )
        
        assert req.encoding_format == "base64"
