"""Tests for Judge scoring system."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from router.judge import JudgeClient, JUDGE_PROMPT_TEMPLATE


class TestJudgeClient:
    """Test JudgeClient functionality."""

    @pytest.fixture
    def judge_enabled(self):
        """Create a JudgeClient with judge enabled."""
        with patch("router.judge.settings") as mock_settings:
            mock_settings.judge_enabled = True
            mock_settings.judge_model = "gpt-4o"
            mock_settings.judge_base_url = "https://api.openai.com/v1"
            mock_settings.judge_api_key = "test-key"
            return JudgeClient()

    @pytest.fixture
    def judge_disabled(self):
        """Create a JudgeClient with judge disabled."""
        with patch("router.judge.settings") as mock_settings:
            mock_settings.judge_enabled = False
            mock_settings.judge_model = "gpt-4o"
            mock_settings.judge_base_url = "https://api.openai.com/v1"
            mock_settings.judge_api_key = None
            return JudgeClient()

    def test_judge_prompt_template(self):
        """Test that judge prompt template has required placeholders."""
        assert "{prompt}" in JUDGE_PROMPT_TEMPLATE
        assert "{response}" in JUDGE_PROMPT_TEMPLATE
        assert "score" in JUDGE_PROMPT_TEMPLATE.lower()

    def test_initialization_enabled(self, judge_enabled):
        """Test JudgeClient initialization with judge enabled."""
        assert judge_enabled.enabled is True
        assert judge_enabled.model == "gpt-4o"

    def test_initialization_disabled(self, judge_disabled):
        """Test JudgeClient initialization with judge disabled."""
        assert judge_disabled.enabled is False

    @pytest.mark.asyncio
    async def test_score_response_disabled(self, judge_disabled):
        """Test scoring when judge is disabled returns fallback."""
        score = await judge_disabled.score_response("Test prompt", "Test response")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_response_empty(self, judge_disabled):
        """Test scoring empty response returns 0."""
        score = await judge_disabled.score_response("Test prompt", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_response_whitespace(self, judge_disabled):
        """Test scoring whitespace-only response returns 0."""
        score = await judge_disabled.score_response("Test prompt", "   ")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_response_short(self, judge_disabled):
        """Test scoring very short response returns fallback when judge disabled."""
        score = await judge_disabled.score_response("Test prompt", "hi")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_response_disabled_with_content(self, judge_disabled):
        """Test scoring non-empty response when disabled returns fallback."""
        score = await judge_disabled.score_response("Test prompt", "This is a valid response")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_response_enabled_success(self, judge_enabled):
        """Test successful scoring with judge enabled."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.85, "reasoning": "Good response"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            score = await judge_enabled.score_response(
                "What is 2+2?",
                "The answer is 4."
            )
            
            assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_response_bounds_high(self, judge_enabled):
        """Test that scores above 1.0 are clamped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 1.5, "reasoning": "Excellent"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            score = await judge_enabled.score_response("Test", "Response")
            
            assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_response_bounds_low(self, judge_enabled):
        """Test that negative scores are clamped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": -0.5, "reasoning": "Poor"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            score = await judge_enabled.score_response("Test", "Response")
            
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_response_missing_score(self, judge_enabled):
        """Test handling missing score field in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"reasoning": "No score provided"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            score = await judge_enabled.score_response("Test", "Response")
            
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_response_api_error(self, judge_enabled):
        """Test fallback when API call fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("API error")
            )
            
            score = await judge_enabled.score_response("Test", "Valid response")
            
            assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_response_invalid_json(self, judge_enabled):
        """Test handling invalid JSON in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "not valid json"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            score = await judge_enabled.score_response("Test", "Response")
            
            assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_response_with_api_key(self, judge_enabled):
        """Test that API key is included in request headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.9})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)
            
            await judge_enabled.score_response("Test", "Response")
            
            call_args = mock_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert "Bearer test-key" in headers["Authorization"]

    @pytest.mark.asyncio
    async def test_score_response_without_api_key(self):
        """Test request works without API key."""
        with patch("router.judge.settings") as mock_settings:
            mock_settings.judge_enabled = True
            mock_settings.judge_model = "local-model"
            mock_settings.judge_base_url = "http://localhost:11434/v1"
            mock_settings.judge_api_key = None
            
            client = JudgeClient()
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [
                    {"message": {"content": json.dumps({"score": 0.7})}}
                ]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = mock_client.return_value.__aenter__.return_value
                mock_instance.post = AsyncMock(return_value=mock_response)
                
                score = await client.score_response("Test", "Response")
                
                assert score == 0.7
