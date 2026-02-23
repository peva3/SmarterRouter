"""Tests for model filtering functionality."""

import pytest
from router.backends.base import ModelInfo
from router.model_filter import (
    filter_model_infos,
    filter_model_names,
    matches_patterns_case_insensitive,
)


class TestMatchesPatternsCaseInsensitive:
    """Test pattern matching with case insensitivity."""

    def test_empty_patterns(self):
        """Empty pattern list returns False."""
        assert matches_patterns_case_insensitive("llama3", []) is False

    def test_single_star_pattern(self):
        """Single * matches everything."""
        assert matches_patterns_case_insensitive("llama3", ["*"]) is True
        assert matches_patterns_case_insensitive("gemma2:7b", ["*"]) is True

    def test_exact_match(self):
        """Exact model name matches."""
        assert matches_patterns_case_insensitive("llama3", ["llama3"]) is True

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert matches_patterns_case_insensitive("LLAMA3", ["llama3"]) is True
        assert matches_patterns_case_insensitive("llama3", ["LLAMA3"]) is True
        assert matches_patterns_case_insensitive("Gemma-3b", ["gemma*"]) is True

    def test_wildcard_star(self):
        """* wildcard matches any characters."""
        assert matches_patterns_case_insensitive("llama3:8b", ["llama*"]) is True
        assert matches_patterns_case_insensitive("llama3.1:70b-instruct", ["llama*"]) is True
        assert matches_patterns_case_insensitive("gemma3:7b", ["*3*"]) is True
        assert matches_patterns_case_insensitive("gemma2:7b", ["*2*"]) is True

    def test_wildcard_question(self):
        """? wildcard matches single character."""
        assert matches_patterns_case_insensitive("llama3", ["llama?"]) is True
        assert matches_patterns_case_insensitive("llama2", ["llama?"]) is True
        assert matches_patterns_case_insensitive("llama10", ["llama??"]) is True

    def test_character_class(self):
        """[seq] matches character in sequence."""
        assert matches_patterns_case_insensitive("llama3", ["llama[123]"]) is True
        assert matches_patterns_case_insensitive("llama2", ["llama[123]"]) is True
        assert matches_patterns_case_insensitive("llama4", ["llama[123]"]) is False

    def test_negated_class(self):
        """[!seq] matches character not in sequence."""
        assert matches_patterns_case_insensitive("llama4", ["llama[!123]"]) is True
        assert matches_patterns_case_insensitive("llama1", ["llama[!123]"]) is False


class TestFilterModelNames:
    """Test filtering model name lists."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert filter_model_names([], [], []) == []
        assert filter_model_names([], ["llama*"], []) == []

    def test_no_filter(self):
        """No filter returns all models."""
        models = ["llama3", "gemma2:7b", "mistral7b"]
        assert filter_model_names(models, [], []) == models

    def test_include_only(self):
        """Include filter only allows matching models."""
        models = ["llama3", "gemma2:7b", "mistral7b", "phi3:mini"]
        
        result = filter_model_names(models, ["llama*", "gemma*"], [])
        assert "llama3" in result
        assert "gemma2:7b" in result
        assert "mistral7b" not in result
        assert "phi3:mini" not in result

    def test_exclude_only(self):
        """Exclude filter removes matching models."""
        models = ["llama3", "qwen2.5:72b", "gemma2:7b", "mistral7b"]
        
        result = filter_model_names(models, [], ["qwen*"])
        assert "llama3" in result
        assert "qwen2.5:72b" not in result
        assert "gemma2:7b" in result
        assert "mistral7b" in result

    def test_exclude_takes_precedence(self):
        """Exclude takes precedence over include."""
        models = ["llama3", "llama3:8b-q4", "gemma2:7b"]
        
        result = filter_model_names(
            models,
            ["llama*"],  # Include llama
            ["*q4*"],    # Exclude q4
        )
        assert "llama3" in result
        assert "llama3:8b-q4" not in result  # Excluded by pattern
        assert "gemma2:7b" not in result  # Not in include

    def test_multiple_exclude_patterns(self):
        """Multiple exclude patterns work together."""
        models = ["llama3", "qwen2.5", "gemma2", "mistral7b"]
        
        result = filter_model_names(models, [], ["qwen*", "mistral*"])
        assert "llama3" in result
        assert "qwen2.5" not in result
        assert "gemma2" in result
        assert "mistral7b" not in result

    def test_case_insensitive_filtering(self):
        """Filtering is case-insensitive."""
        models = ["LLAMA3", "Gemma2:7b", "Mistral7b"]
        
        result = filter_model_names(models, ["llama*", "gemma*"], [])
        assert "LLAMA3" in result
        assert "Gemma2:7b" in result
        assert "Mistral7b" not in result


class TestFilterModelInfos:
    """Test filtering ModelInfo objects."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert filter_model_infos([], [], []) == []

    def test_no_filter(self):
        """No filter returns all models."""
        models = [
            ModelInfo(name="llama3", size=1234567890),
            ModelInfo(name="gemma2:7b", size=987654321),
        ]
        result = filter_model_infos(models, [], [])
        assert len(result) == 2
        assert result[0].name == "llama3"
        assert result[1].name == "gemma2:7b"

    def test_include_filter(self):
        """Include filter works with ModelInfo."""
        models = [
            ModelInfo(name="llama3", size=1234567890),
            ModelInfo(name="qwen2.5:72b", size=40000000000),
            ModelInfo(name="gemma2:7b", size=987654321),
        ]
        
        result = filter_model_infos(models, ["llama*", "gemma*"], [])
        assert len(result) == 2
        names = [m.name for m in result]
        assert "llama3" in names
        assert "gemma2:7b" in names
        assert "qwen2.5:72b" not in names

    def test_exclude_filter(self):
        """Exclude filter works with ModelInfo."""
        models = [
            ModelInfo(name="llama3", size=1234567890),
            ModelInfo(name="qwen2.5:72b", size=40000000000),
            ModelInfo(name="gemma2:7b", size=987654321),
        ]
        
        result = filter_model_infos(models, [], ["qwen*"])
        assert len(result) == 2
        names = [m.name for m in result]
        assert "llama3" in names
        assert "gemma2:7b" in names
        assert "qwen2.5:72b" not in names

    def test_preserves_model_info(self):
        """Filtered results preserve all ModelInfo fields."""
        models = [
            ModelInfo(name="llama3", size=1234567890, modified_at="2024-01-01"),
        ]
        
        result = filter_model_infos(models, ["llama*"], [])
        assert len(result) == 1
        assert result[0].name == "llama3"
        assert result[0].size == 1234567890
        assert result[0].modified_at == "2024-01-01"


class TestFilterEdgeCases:
    """Test edge cases in filtering."""

    def test_all_excluded(self):
        """All models excluded returns empty list."""
        models = ["llama3", "gemma2:7b", "mistral7b"]
        
        result = filter_model_names(models, [], ["*"])
        assert result == []

    def test_nothing_matches_include(self):
        """No models match include returns empty."""
        models = ["llama3", "gemma2:7b"]
        
        result = filter_model_names(models, ["qwen*"], [])
        assert result == []

    def test_overlapping_include_exclude(self):
        """Overlapping include/exclude patterns work correctly."""
        models = ["llama3", "llama3:8b-q4", "llama3:8b-q8"]
        
        # Include all llama, but exclude q4
        result = filter_model_names(models, ["llama3*"], ["*q4*"])
        assert "llama3" in result
        assert "llama3:8b-q4" not in result
        assert "llama3:8b-q8" in result

    def test_special_characters_in_names(self):
        """Model names with special characters are handled correctly."""
        models = ["model:v1-q4_K_M", "model:v2-q5_K_M", "other"]
        
        result = filter_model_names(models, [], ["*q4*"])
        assert "model:v1-q4_K_M" not in result
        assert "model:v2-q5_K_M" in result
        assert "other" in result
