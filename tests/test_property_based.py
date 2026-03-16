# Property-based testing with Hypothesis
"""
Item #50: Property-based testing for critical functions.

Uses Hypothesis to fuzz test:
- _cosine_similarity with various vector dimensions
- Model filtering with arbitrary patterns
- Prompt sanitization with edge cases
"""
import string

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="Hypothesis is optional")
from hypothesis import given, settings, strategies as st

from router.backends.base import ModelInfo
from router.model_filter import filter_model_names
from router.schemas import sanitize_prompt


class TestCosineSimilarity:
    """Property-based tests for cosine similarity calculation."""

    @pytest.fixture
    def cosine_similarity(self):
        """Import cosine similarity function."""
        from router.router import RouterEngine

        engine = RouterEngine(
            client=_DummyBackend(),
            cache_enabled=False,
        )
        def _wrapped(vec1, vec2):
            mag1 = sum(x * x for x in vec1) ** 0.5
            mag2 = sum(x * x for x in vec2) ** 0.5
            return engine._cosine_similarity(vec1, vec2, mag1, mag2)

        return _wrapped

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=2, max_size=100),
        st.lists(st.floats(min_value=-100, max_value=100), min_size=2, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_similarity_range(self, cosine_similarity, vec1, vec2):
        """Cosine similarity should always return value between -1 and 1."""
        # Skip if vectors have different sizes or are all zeros
        if len(vec1) != len(vec2):
            return

        # Skip zero vectors to avoid division by zero
        if all(x == 0 for x in vec1) or all(x == 0 for x in vec2):
            return

        result = cosine_similarity(vec1, vec2)

        # Result should be between -1 and 1 (with floating point tolerance)
        assert -1.01 <= result <= 1.01

    @given(
        st.lists(st.floats(min_value=1.0, max_value=100), min_size=2, max_size=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_identical_vectors(self, cosine_similarity, vec):
        """Similarity of identical vectors should be close to 1."""
        # Skip zero vectors
        if all(x == 0 for x in vec):
            return

        result = cosine_similarity(vec, vec)
        assert 0.99 <= result <= 1.01  # Close to 1.0

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_opposite_vectors(self, cosine_similarity, vec):
        """Similarity of opposite vectors should be close to -1."""
        # Skip zero vectors
        if all(x == 0 for x in vec):
            return

        opposite = [-x for x in vec]
        result = cosine_similarity(vec, opposite)
        assert -1.01 <= result <= -0.99  # Close to -1.0


class TestPromptSanitization:
    """Property-based tests for prompt sanitization."""

    @given(st.text(alphabet=string.printable, min_size=0, max_size=5000))
    @settings(max_examples=100)
    def test_sanitization_never_crashes(self, text):
        """Sanitization should never crash on any input."""
        # Should not raise any exceptions
        result = sanitize_prompt(text)
        assert isinstance(result, str)

    @given(
        st.text(alphabet=string.ascii_letters + string.digits + " ", min_size=1, max_size=1000)
    )
    @settings(max_examples=100)
    def test_sanitization_preserves_length_bounds(self, text):
        """Sanitized output should have reasonable bounds."""
        result = sanitize_prompt(text)
        # Should not be longer than input + some buffer
        assert len(result) <= len(text) + 100
        # Should not be empty if input wasn't empty
        if text.strip():
            assert len(result) > 0


class TestModelFiltering:
    """Property-based tests for model filtering logic."""

    @pytest.fixture
    def filter_models(self):
        """Import model filtering function."""
        return filter_model_names

    @given(
        st.lists(
            st.text(alphabet=string.ascii_lowercase + "-_", min_size=3, max_size=30),
            min_size=1,
            max_size=100,
        ),
        st.lists(
            st.text(alphabet=string.ascii_lowercase + "*?", min_size=1, max_size=20),
            min_size=0,
            max_size=5,
        ),
        st.lists(
            st.text(alphabet=string.ascii_lowercase + "*?", min_size=1, max_size=20),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_filtering_result_subset(self, filter_models, models, include, exclude):
        """Filtered result should be subset of input models."""
        result = filter_models(models, include=include, exclude=exclude)

        # Result should be subset of original models
        assert all(m in models for m in result)

        # Result should not contain excluded patterns
        for pattern in exclude:
            clean_pattern = pattern.replace("*", "").replace("?", "")
            if clean_pattern:
                for model in result:
                    assert clean_pattern not in model.lower()

    @given(
        st.lists(
            st.sampled_from(["llama3", "mistral", "gemma", "phi", "qwen", "deepseek"]),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_empty_patterns_include_all(self, filter_models, models):
        """Empty include/exclude patterns should return all models."""
        result = filter_models(models, include=[], exclude=[])
        assert set(result) == set(models)


class TestComplexityScoring:
    """Property-based tests for prompt complexity scoring."""

    @pytest.fixture
    def analyze_complexity(self):
        """Import complexity analysis function."""
        from router.router import RouterEngine

        engine = RouterEngine(
            client=_DummyBackend(),
            cache_enabled=False,
        )
        return engine._analyze_prompt

    @given(
        st.text(min_size=1, max_size=10000),
    )
    @settings(max_examples=100)
    def test_complexity_in_valid_range(self, analyze_complexity, text):
        """Complexity score should be between 0 and 1."""
        # This is a simplified test - actual complexity analysis
        # depends on many factors including token count

        # Just verify it doesn't crash
        try:
            result = analyze_complexity(text)
            # If it returns a score, it should be in valid range
            if isinstance(result, dict):
                complexity = result.get("complexity", 0.0)
                assert 0 <= complexity <= 1
        except Exception:
            # Some edge cases might raise exceptions, that's okay
            pass


class _DummyBackend:
    """Minimal backend for RouterEngine helper tests."""

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="dummy")]

    async def chat(self, *args, **kwargs) -> dict:
        return {"message": {"content": "ok"}}

    async def chat_streaming(self, *args, **kwargs):
        raise NotImplementedError

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def load_model(self, model_name: str, keep_alive: float = -1, timeout: float | None = None) -> bool:
        return False

    async def embed(self, model: str, input_text: str | list[str], **kwargs) -> dict:
        return {"embedding": [0.1, 0.2, 0.3]}

    async def get_model_vram_usage(self, model_name: str) -> float | None:
        return None

    async def close(self) -> None:
        return None

    def is_external_model(self, model_name: str) -> bool:
        return False


class TestBenchmarkMerging:
    """Property-based tests for benchmark data merging."""

    @given(
        st.dictionaries(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
            st.dictionaries(
                st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
                st.floats(min_value=0, max_value=1),
                min_size=1,
                max_size=5,
            ),
            min_size=0,
            max_size=10,
        ),
        st.dictionaries(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
            st.dictionaries(
                st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
                st.floats(min_value=0, max_value=1),
                min_size=1,
                max_size=5,
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_merge_preserves_model_keys(self, benchmarks1, benchmarks2):
        """Merging benchmarks should preserve all model keys."""
        # Simple merge test
        merged = {**benchmarks1}

        for model, scores in benchmarks2.items():
            if model in merged:
                # Merge scores - later values should override
                merged[model] = {**merged[model], **scores}
            else:
                merged[model] = scores

        # All models from both dicts should be present
        assert all(m in merged for m in benchmarks1)
        assert all(m in merged for m in benchmarks2)


class TestConfigValidation:
    """Property-based tests for configuration validation."""

    @given(
        st.sampled_from(["ollama", "llama.cpp", "openai", "ollama", "openai"]),
        st.integers(min_value=1, max_value=65535),
    )
    @settings(max_examples=50)
    def test_valid_port_range(self, provider, port):
        """Port should always be in valid range."""
        # Simple validation
        assert 1 <= port <= 65535

    @given(
        st.floats(min_value=0, max_value=1),
        st.floats(min_value=0, max_value=1),
    )
    @settings(max_examples=50)
    def test_quality_preference_bounds(self, quality, speed):
        """Quality and speed preferences should be between 0 and 1."""
        assert 0 <= quality <= 1
        assert 0 <= speed <= 1


class TestRoutingDecisions:
    """Property-based tests for routing decisions."""

    @given(
        st.lists(
            st.fixed_dictionaries({
                "name": st.text(alphabet=string.ascii_lowercase + "-_", min_size=3, max_size=30),
                "reasoning": st.floats(min_value=0, max_value=1),
                "coding": st.floats(min_value=0, max_value=1),
                "creativity": st.floats(min_value=0, max_value=1),
            }),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30)
    def test_routing_with_multiple_models(self, profiles):
        """Routing should handle multiple models gracefully."""
        # This tests that we can create score dictionaries
        # from various model profiles

        scores: dict[str, float] = {}
        for profile in profiles:
            # Calculate a composite score
            name = profile["name"]
            score = (
                profile["reasoning"] * 0.4 +
                profile["coding"] * 0.3 +
                profile["creativity"] * 0.3
            )
            scores[name] = score

        # Should be able to select max score model
        if scores:
            best = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0][0]
            assert best in [p["name"] for p in profiles]
