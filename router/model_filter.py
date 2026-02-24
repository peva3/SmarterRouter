"""Model filtering utilities for SmarterRouter.

Provides glob-based filtering for model names with include/exclude patterns.
Uses case-insensitive matching for user convenience.

Pattern Syntax:
    *       Matches everything
    ?       Matches any single character
    [seq]   Matches any character in seq
    [!seq]  Matches any character not in seq

Examples:
    include=["gemma*", "mistral*"], exclude=["*qwen*"]
    - Allows gemma-*, mistral-* models
    - Blocks any model with "qwen" in name
    - Case-insensitive: "Gemma", "GEMMA" both match "gemma*"
"""

import fnmatch
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.backends.base import ModelInfo

logger = logging.getLogger(__name__)


def matches_patterns_case_insensitive(name: str, patterns: list[str]) -> bool:
    """Check if name matches any of the glob patterns (case-insensitive).

    Args:
        name: Model name to check
        patterns: List of glob patterns

    Returns:
        True if name matches any pattern, False otherwise
    """
    if not patterns:
        return False
    name_lower = name.lower()
    return any(fnmatch.fnmatch(name_lower, pat.lower()) for pat in patterns)


def filter_model_infos(
    models: list["ModelInfo"], include: list[str], exclude: list[str]
) -> list["ModelInfo"]:
    """Filter list of ModelInfo based on include/exclude patterns.

    Filtering logic:
        1. If include is empty, all models are candidates
        2. If include is not empty, only models matching include patterns are candidates
        3. Exclude patterns are always applied after include
        4. Exclude takes precedence: excluded models are removed regardless of include

    Args:
        models: List of ModelInfo objects to filter
        include: List of glob patterns to include (empty = include all)
        exclude: List of glob patterns to exclude (empty = exclude none)

    Returns:
        Filtered list of ModelInfo objects
    """
    if not models:
        return []

    filtered: list[ModelInfo] = []

    for model in models:
        # First check exclude patterns (always applied)
        if matches_patterns_case_insensitive(model.name, exclude):
            logger.debug(f"Model '{model.name}' excluded by filter pattern")
            continue

        # Then check include patterns (if specified)
        if include and not matches_patterns_case_insensitive(model.name, include):
            logger.debug(f"Model '{model.name}' not in include patterns")
            continue

        filtered.append(model)

    return filtered


def filter_model_names(names: list[str], include: list[str], exclude: list[str]) -> list[str]:
    """Filter list of model names based on include/exclude patterns.

    Same filtering logic as filter_model_infos but operates on string names.

    Args:
        names: List of model names to filter
        include: List of glob patterns to include (empty = include all)
        exclude: List of glob patterns to exclude (empty = exclude none)

    Returns:
        Filtered list of model names
    """
    if not names:
        return []

    filtered: list[str] = []

    for name in names:
        # First check exclude patterns (always applied)
        if matches_patterns_case_insensitive(name, exclude):
            logger.debug(f"Model '{name}' excluded by filter pattern")
            continue

        # Then check include patterns (if specified)
        if include and not matches_patterns_case_insensitive(name, include):
            logger.debug(f"Model '{name}' not in include patterns")
            continue

        filtered.append(name)

    return filtered


def log_filter_summary(
    total_models: int,
    included_count: int,
    excluded_by_pattern: int,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> None:
    """Log a summary of model filtering results.

    Args:
        total_models: Total models before filtering
        included_count: Models that passed filtering
        excluded_by_pattern: Models excluded by patterns
        include_patterns: Include patterns used
        exclude_patterns: Exclude patterns used
    """
    if include_patterns or exclude_patterns:
        logger.info(
            f"Model filter: {total_models} total, {included_count} included, "
            f"{excluded_by_pattern} excluded "
            f"(include={include_patterns}, exclude={exclude_patterns})"
        )
