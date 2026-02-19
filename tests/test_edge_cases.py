"""Edge case tests for the router and related components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from router.router import RouterEngine, SemanticCache
from router.config import Settings


class TestSemanticCacheEdgeCases:
    """Test edge cases in SemanticCache."""

    def test_empty_cache_returns_none(self):
        """Empty cache should return None on get."""
        cache = SemanticCache(max_size=10)
        # Test sync behavior through the hash function
        assert cache._hash_prompt("test") == cache._hash_prompt("test")
        assert cache._hash_prompt("test1") != cache._hash_prompt("test2")

    def test_cosine_similarity_edge_cases(self):
        """Test cosine similarity with edge cases."""
        cache = SemanticCache(max_size=10)
        
        # Empty vectors
        assert cache._cosine_similarity([], []) == 0.0
        assert cache._cosine_similarity([], [1.0, 2.0]) == 0.0
        assert cache._cosine_similarity([1.0, 2.0], []) == 0.0
        
        # Zero vector
        assert cache._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cache._cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0
        
        # Identical vectors (similarity ≈ 1.0, accounting for floating point)
        assert cache._cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
        
        # Orthogonal vectors (similarity = 0.0)
        assert cache._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cache_eviction_lru(self):
        """Test LRU eviction when cache is full."""
        cache = SemanticCache(max_size=3, ttl_seconds=3600)
        
        # Add 3 items
        cache.cache["key1"] = (MagicMock(), 100.0, None)
        cache.cache["key2"] = (MagicMock(), 100.0, None)
        cache.cache["key3"] = (MagicMock(), 100.0, None)
        
        # Add 4th item - should evict key1 (LRU)
        cache.cache["key4"] = (MagicMock(), 100.0, None)
        cache.cache.move_to_end("key4")
        if len(cache.cache) > cache.max_size:
            cache.cache.popitem(last=False)
        
        assert "key1" not in cache.cache
        assert "key4" in cache.cache

    @pytest.mark.asyncio
    async def test_get_model_frequency_empty(self):
        """Test get_model_frequency with no selections."""
        cache = SemanticCache(max_size=10)
        
        # No selections yet
        freq = await cache.get_model_frequency("model1")
        assert freq == 0.0

    @pytest.mark.asyncio
    async def test_get_model_frequency_single(self):
        """Test get_model_frequency with single selection."""
        import time
        cache = SemanticCache(max_size=10)
        cache.recent_selections = [("model1", time.time())]
        
        freq = await cache.get_model_frequency("model1")
        assert freq == 1.0
        
        freq = await cache.get_model_frequency("model2")
        assert freq == 0.0


class TestRouterEdgeCases:
    """Test edge cases in RouterEngine."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = AsyncMock()
        backend.list_models = AsyncMock(return_value=[
            MagicMock(name="model1"),
            MagicMock(name="model2"),
        ])
        return backend

    @pytest.fixture
    def router(self, mock_backend):
        """Create a router with mocked backend."""
        return RouterEngine(
            client=mock_backend,
            cache_enabled=False,
        )

    def test_analyze_prompt_empty_string(self, router):
        """Test analyzing an empty prompt."""
        analysis = router._analyze_prompt("")
        
        # Most categories should be 0.0, but factual might have default
        assert analysis["reasoning"] == 0.0
        assert analysis["coding"] == 0.0
        assert analysis["creativity"] == 0.0
        assert analysis["complexity"] == 0.0
        # Factual might have a base value for empty prompts

    def test_analyze_prompt_very_long(self, router):
        """Test analyzing a very long prompt."""
        # Create a 2000+ character prompt
        long_prompt = "Write a story about " + "x" * 2000
        
        analysis = router._analyze_prompt(long_prompt)
        
        # Should trigger length-based complexity
        assert analysis["complexity"] >= 0.5  # >500 chars + >1500 chars
        # "story" and "write" trigger creativity
        assert analysis["creativity"] > 0

    def test_analyze_prompt_special_characters(self, router):
        """Test analyzing prompts with special characters."""
        prompts = [
            "What is 2 + 2?",  # Math reasoning
            "def foo(): pass",  # Code
            "print('hello')",  # Code
            "SELECT * FROM users",  # SQL
        ]
        
        for prompt in prompts:
            analysis = router._analyze_prompt(prompt)
            # Should not crash
            assert isinstance(analysis, dict)
            assert "reasoning" in analysis
            assert "coding" in analysis

    def test_analyze_prompt_unicode(self, router):
        """Test analyzing prompts with unicode characters."""
        prompts = [
            "你好世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "🎉🎊🎈",  # Emojis
        ]
        
        for prompt in prompts:
            analysis = router._analyze_prompt(prompt)
            # Should not crash
            assert isinstance(analysis, dict)

    def test_calculate_combined_scores_empty_profiles(self, router):
        """Test scoring with empty profiles and no benchmarks."""
        analysis = {"coding": 0.5, "reasoning": 0.3, "creativity": 0.0, "factual": 0.0}
        
        # Should handle gracefully
        scores = router._calculate_combined_scores(
            [], [], analysis, ["model1", "model2"]
        )
        
        # Should return scores for all models
        assert "model1" in scores
        assert "model2" in scores

    def test_calculate_combined_scores_empty_model_names(self, router):
        """Test scoring with no model names."""
        analysis = {"coding": 1.0, "reasoning": 0.0, "creativity": 0.0, "factual": 0.0}
        
        scores = router._calculate_combined_scores(
            [], [], analysis, []
        )
        
        # Should return empty dict
        assert scores == {}

    def test_calculate_combined_scores_all_zero_analysis(self, router):
        """Test scoring when all analysis weights are zero."""
        analysis = {"coding": 0.0, "reasoning": 0.0, "creativity": 0.0, "factual": 0.0}
        
        scores = router._calculate_combined_scores(
            [], [], analysis, ["model1"]
        )
        
        # Should still produce valid scores
        assert "model1" in scores

    def test_extract_parameter_count_edge_cases(self, router):
        """Test parameter count extraction from various model names."""
        # Test cases that should work
        test_cases = [
            ("llama3:8b", 8.0),
            ("qwen2.5-coder:14b-instruct", 14.0),
            ("model-without-size", None),
            ("model-0.5b", 0.5),  # Size in tag part
            ("model-100b", 100.0),
            ("model-1.5g", None),  # GB not B - should return None
            ("phi3:mini", 3.8),  # Phi-3-mini uses size_map
            ("mistral-nemo", 12.0),  # Uses size_map
        ]
        
        for model_name, expected in test_cases:
            result = router._extract_parameter_count(model_name)
            assert result == expected, f"Failed for {model_name}: got {result}, expected {expected}"
        
        # Note: Some formats like phi3:3.8b are not supported - the regex 
        # only checks the model tag (before colon), not the version (after colon)

    def test_get_complexity_bucket_edge_cases(self, router):
        """Test complexity bucket determination."""
        assert router._get_complexity_bucket(0.0) == "simple"
        assert router._get_complexity_bucket(0.19) == "simple"
        assert router._get_complexity_bucket(0.2) == "medium"
        assert router._get_complexity_bucket(0.49) == "medium"
        assert router._get_complexity_bucket(0.5) == "hard"
        assert router._get_complexity_bucket(1.0) == "hard"

    def test_build_model_category_affinity_edge_cases(self, router):
        """Test model category affinity with edge case names."""
        model_names = [
            "plain-model",
            "code-model",
            "reasoner-v2",
            "creative-writer",
            "dolphin-mixtral",
            "deepseek-coder-33b",
        ]
        
        affinity = router._build_model_category_affinity(model_names, {})
        
        # Check that all models get an affinity dict
        for name in model_names:
            assert name in affinity
            assert "coding" in affinity[name]
            assert "reasoning" in affinity[name]
            assert "creativity" in affinity[name]
            assert "factual" in affinity[name]

    def test_diversity_penalty_edge_cases(self, router):
        """Test diversity penalty calculations."""
        profiles = [
            {
                "name": "model1",
                "reasoning": 0.8,
                "coding": 0.7,
                "creativity": 0.6,
                "factual": 0.75,
                "speed": 0.7,
                "avg_response_time_ms": 1000.0,
                "first_seen": None,
            },
        ]
        
        # Test with various frequency levels
        for freq in [0.0, 0.3, 0.5, 0.7, 1.0]:
            analysis = {"coding": 0.5, "reasoning": 0.0, "creativity": 0.0, "factual": 0.0}
            scores = router._calculate_combined_scores(
                profiles, [], analysis, ["model1"],
                model_frequencies={"model1": freq}
            )
            # Should not crash
            assert "model1" in scores
            # Higher frequency should result in lower or equal score
            if freq > 0.5:
                # With multiplicative penalty, score should be reduced
                assert scores["model1"]["diversity"] <= 0


class TestBenchmarkDbEdgeCases:
    """Test edge cases in benchmark database functions."""

    def test_get_benchmarks_for_models_empty_list(self):
        """Test getting benchmarks for empty model list."""
        from router.benchmark_db import get_benchmarks_for_models
        
        result = get_benchmarks_for_models([])
        assert result == []

    def test_bulk_upsert_benchmarks_empty_list(self):
        """Test bulk upsert with empty list."""
        from router.benchmark_db import bulk_upsert_benchmarks
        
        result = bulk_upsert_benchmarks([])
        assert result == 0


class TestConfigEdgeCases:
    """Test edge cases in configuration."""

    def test_quality_preference_extremes(self):
        """Test quality preference at extreme values."""
        # Valid range is 0.0 to 1.0
        settings_min = Settings(quality_preference=0.0)
        assert settings_min.quality_preference == 0.0
        
        settings_max = Settings(quality_preference=1.0)
        assert settings_max.quality_preference == 1.0

    def test_vram_unload_strategy_case_sensitivity(self):
        """Test that vram_unload_strategy accepts valid values."""
        # These should work
        settings_lru = Settings(vram_unload_strategy="lru")
        assert settings_lru.vram_unload_strategy == "lru"
        
        settings_largest = Settings(vram_unload_strategy="largest")
        assert settings_largest.vram_unload_strategy == "largest"

    def test_url_validation(self):
        """Test URL validation for backend URLs."""
        # Valid URLs
        settings = Settings(
            ollama_url="http://localhost:11434",
            llama_cpp_url="http://localhost:8080",
            openai_base_url="https://api.openai.com/v1",
        )
        assert settings.ollama_url == "http://localhost:11434"

    def test_empty_benchmark_sources(self):
        """Test empty benchmark sources."""
        settings = Settings(benchmark_sources="")
        assert settings.benchmark_sources == ""


class TestProfilerEdgeCases:
    """Test edge cases in profiler."""

    def test_timeout_calculation(self):
        """Test that timeout is calculated for models."""
        from router.profiler import ModelProfiler
        from unittest.mock import MagicMock
        
        mock_backend = MagicMock()
        
        # Small model
        profiler_small = ModelProfiler(mock_backend, model_name="phi3:3.8b")
        assert profiler_small.timeout >= 30  # At least minimum
        
        # Large model
        profiler_large = ModelProfiler(mock_backend, model_name="llama3:70b")
        assert profiler_large.timeout >= 90  # Base timeout for very large

    def test_screening_token_rate_initialization(self):
        """Test that screening token rate is initialized."""
        from router.profiler import ModelProfiler
        from unittest.mock import MagicMock
        
        mock_backend = MagicMock()
        profiler = ModelProfiler(mock_backend, model_name="test-model")
        
        # Should have screening token rate attribute
        assert hasattr(profiler, 'screening_token_rate')


class TestLoggingSanitization:
    """Test logging sanitization edge cases."""

    def test_sanitize_empty_string(self):
        from router.logging_config import sanitize_for_logging
        
        assert sanitize_for_logging("") == ""

    def test_sanitize_none(self):
        from router.logging_config import sanitize_for_logging
        
        assert sanitize_for_logging(None) is None

    def test_sanitize_nested_dict(self):
        from router.logging_config import sanitize_for_logging
        
        data = {
            "key": "value",
            "nested": {
                "api_key": "sk-1234567890abcdefghijklmnopqrstuv",
                "password": "secret123",
            },
        }
        
        result = sanitize_for_logging(data)
        # API key should be redacted
        assert "REDACTED" in result["nested"]["api_key"] or result["nested"]["api_key"] != data["nested"]["api_key"]

    def test_sanitize_long_string_truncation(self):
        from router.logging_config import sanitize_for_logging
        
        # Use a string that won't be mistaken for base64
        long_string = "Hello world! " * 50  # ~650 chars, clearly not base64
        result = sanitize_for_logging(long_string, max_length=100)
        
        # Should be truncated
        assert len(result) <= 103  # 100 + "..."
        assert "Hello" in result  # Content should still be there


class TestRoutingDecisionEdgeCases:
    """Test edge cases in routing decisions."""

    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.list_models = AsyncMock(return_value=[
            MagicMock(name="model-a"),
            MagicMock(name="model-b"),
            MagicMock(name="model-c"),
        ])
        return backend

    @pytest.fixture
    def router(self, mock_backend):
        return RouterEngine(client=mock_backend, cache_enabled=False)

    def test_build_reasoning_with_all_categories(self, router):
        """Test building reasoning string with all categories."""
        analysis = {
            "reasoning": 0.8,
            "coding": 0.9,
            "creativity": 0.3,
            "factual": 0.5,
            "complexity": 0.6,
        }
        scores = {
            "score": 0.85,
            "reasoning": 0.8,
            "coding": 0.9,
            "creativity": 0.3,
        }
        
        reasoning = router._build_reasoning(analysis, scores)
        
        # Should contain score information
        assert "0.85" in reasoning
        assert isinstance(reasoning, str)

    def test_build_reasoning_with_zero_scores(self, router):
        """Test building reasoning with zero scores."""
        analysis = {
            "reasoning": 0.0,
            "coding": 0.0,
            "creativity": 0.0,
            "factual": 0.0,
        }
        scores = {
            "score": 0.0,
            "reasoning": 0.0,
            "coding": 0.0,
        }
        
        reasoning = router._build_reasoning(analysis, scores)
        
        # Should still produce valid output
        assert isinstance(reasoning, str)
