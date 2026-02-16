from unittest.mock import AsyncMock, patch

import pytest

from router.router import RouterEngine


@pytest.fixture
def mock_client():
    client = AsyncMock()
    return client


@pytest.fixture
def router(mock_client):
    return RouterEngine(mock_client, dispatcher_model=None)


@pytest.fixture
def sample_profiles():
    return [
        {
            "name": "llama3",
            "reasoning": 0.9,
            "coding": 0.7,
            "creativity": 0.8,
            "factual": 0.85,
            "speed": 0.6,
            "avg_response_time_ms": 2000.0,
            "first_seen": None,
        },
        {
            "name": "codellama",
            "reasoning": 0.7,
            "coding": 0.95,
            "creativity": 0.6,
            "factual": 0.75,
            "speed": 0.5,
            "avg_response_time_ms": 3000.0,
            "first_seen": None,
        },
        {
            "name": "mistral",
            "reasoning": 0.8,
            "coding": 0.6,
            "creativity": 0.7,
            "factual": 0.8,
            "speed": 0.85,
            "avg_response_time_ms": 800.0,
            "first_seen": None,
        },
    ]


@pytest.mark.asyncio
async def test_select_model_coding_prompt(router, sample_profiles):
    """When no benchmarks available, algorithm prefers faster models.
    Mistral has highest speed (0.85), so it wins even for coding prompts.
    This test verifies the speed-based fallback behavior."""
    with patch.object(router, "_get_all_profiles", return_value=sample_profiles):
        with patch("router.router.get_all_benchmarks", return_value=[]):
            result = await router._keyword_dispatch(
                "Write a Python function to calculate fibonacci",
                ["llama3", "codellama", "mistral"]
            )

            # Without benchmarks, speed bonus determines winner
            # mistral has highest speed (0.85), so it wins
            assert result.selected_model == "mistral"
            assert result.confidence > 0


@pytest.mark.asyncio
async def test_select_model_reasoning_prompt(router, sample_profiles):
    """When no benchmarks available, algorithm prefers faster models.
    Mistral has highest speed (0.85), so it wins even for reasoning prompts."""
    with patch.object(router, "_get_all_profiles", return_value=sample_profiles):
        with patch("router.router.get_all_benchmarks", return_value=[]):
            result = await router._keyword_dispatch(
                "If a train travels 120km in 1.5 hours, what is its average speed in m/s?",
                ["llama3", "codellama", "mistral"]
            )

            # Without benchmarks, speed bonus determines winner
            assert result.selected_model == "mistral"
            assert result.confidence > 0


@pytest.mark.asyncio
async def test_select_model_fast_for_simple_prompt(router, sample_profiles):
    with patch.object(router, "_get_all_profiles", return_value=sample_profiles):
        with patch("router.router.get_all_benchmarks", return_value=[]):
            result = await router._keyword_dispatch(
                "What is the capital of France?",
                ["llama3", "codellama", "mistral"]
            )

            assert result.selected_model in ["llama3", "mistral"]


def test_analyze_prompt_coding(router):
    analysis = router._analyze_prompt(
        "Write a Python function to check if a string is a palindrome"
    )

    assert analysis["coding"] > analysis["reasoning"]
    assert analysis["coding"] > analysis["factual"]


def test_analyze_prompt_creative(router):
    analysis = router._analyze_prompt("Write a short story about a robot")

    assert analysis["creativity"] > 0


def test_analyze_prompt_fallback(router):
    analysis = router._analyze_prompt("Hello, how are you?")

    assert max(analysis.values()) > 0


def test_calculate_combined_scores_with_profiles(router, sample_profiles):
    """Test scoring with profiles but no benchmarks.
    Without benchmarks, speed is the main differentiator.
    mistral has highest speed (0.85), so it scores highest."""
    analysis = {"coding": 1.0, "reasoning": 0.0, "creativity": 0.0, "factual": 0.0}

    scores = router._calculate_combined_scores(
        sample_profiles, [], analysis, ["llama3", "codellama", "mistral"]
    )

    # Without benchmarks, speed bonus determines scores
    # mistral has highest speed, so it scores highest
    assert scores["mistral"]["score"] > scores["llama3"]["score"]


def test_build_reasoning(router):
    analysis = {"reasoning": 1.0, "coding": 0.0, "creativity": 0.0, "factual": 0.0}
    scores = {"score": 0.85, "reasoning": 0.9, "coding": 0.7}

    reasoning = router._build_reasoning(analysis, scores)

    assert "reasoning" in reasoning
    assert "0.85" in reasoning
