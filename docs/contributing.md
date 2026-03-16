# Developer Contribution Guide

Thank you for your interest in contributing to SmarterRouter! This guide covers everything you need to know to contribute effectively.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Triage](#issue-triage)

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/smarterrouter.git
cd smarterrouter
```

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

## Development Environment

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_router.py

# Run specific test
pytest tests/test_router.py::test_select_best_model

# Run with verbose output
pytest -v -s
```

### Run the Server

```bash
# Development mode
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436

# With custom config
ROUTER_LOG_LEVEL=DEBUG python -m uvicorn main:app --reload
```

### Code Quality Checks

```bash
# Lint with ruff
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Type check
mypy .

# Run all checks
ruff check . && ruff format . && mypy .
```

## Coding Standards

### Python Style

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Standard library first, then third-party, then local
- **Types**: Use explicit type hints for all function signatures

Example:

```python
from typing import Any

import httpx
from fastapi import FastAPI

from router.config import Settings


def process_items(items: list[str]) -> dict[str, int]:
    """Process items and return counts.
    
    Args:
        items: List of items to process
        
    Returns:
        Dictionary mapping items to their counts
    """
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables | snake_case | `model_name`, `response_text` |
| Functions | snake_case | `get_best_model()` |
| Classes | PascalCase | `ModelProfiler` |
| Constants | SCREAMING_SNAKE | `MAX_RETRIES` |
| Private methods | _snake_case | `_fetch_models()` |

### Docstrings

Use Google/NumPy style docstrings:

```python
def calculate_score(
    model: ModelProfile,
    analysis: PromptAnalysis,
    feedback: dict[str, float],
) -> float:
    """Calculate routing score for a model.
    
    Combines benchmark scores, profile scores, and user feedback
    to determine the best model for a given prompt.
    
    Args:
        model: Model profile with capabilities
        analysis: Analyzed prompt characteristics
        feedback: User feedback scores by model
        
    Returns:
        Float score between 0.0 and 1.0
        
    Raises:
        ValueError: If model or analysis is invalid
    """
    if not model or not analysis:
        raise ValueError("Model and analysis required")
    
    # Implementation
    return score
```

### Error Handling

```python
class RouterError(Exception):
    """Base exception for routing errors."""
    pass


def fetch_data(url: str) -> dict[str, Any]:
    """Fetch data from URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
        raise RouterError(f"Timeout fetching data") from None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}")
        raise RouterError(f"HTTP error: {e.response.status_code}") from None
```

### Async Patterns

```python
# Use async/await consistently
async def fetch_models() -> list[str]:
    """Fetch available models."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/tags")
        return response.json()["models"]


# Parallel operations with gather
async def profile_models(models: list[str]) -> dict[str, Profile]:
    """Profile multiple models in parallel."""
    tasks = [profile_model(m) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        m: r for m, r in zip(models, results)
        if not isinstance(r, Exception)
    }
```

## Testing

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_backend():
    """Create mock backend fixture."""
    return AsyncMock()


@pytest.mark.asyncio
async def test_model_selection_simple(mock_backend):
    """Test basic model selection."""
    # Arrange
    profiles = [
        ModelProfile(name="model1", reasoning=0.9),
        ModelProfile(name="model2", reasoning=0.7),
    ]
    prompt = "What is 2+2?"
    
    # Act
    result = await select_model(prompt, profiles)
    
    # Assert
    assert result == "model1"
```

### Test Guidelines

1. **Use descriptive names**: `test_<method>_<expected_behavior>`
2. **One assertion per test** (generally)
3. **Mock external dependencies** (HTTP, database)
4. **Test edge cases**: empty inputs, None values, large inputs
5. **Use pytest-asyncio** for async tests

### Coverage Goals

- Minimum 90% code coverage
- 100% coverage for critical paths (routing, security)
- Focus on unit tests; integration tests for key workflows

## Pull Request Process

### Before Submitting

1. **Run all tests**: `pytest`
2. **Check coverage**: `pytest --cov=.`
3. **Lint and format**: `ruff check . && ruff format .`
4. **Type check**: `mypy .`
5. **Update documentation** if needed

### PR Guidelines

1. **One feature/fix per PR**
2. **Clear description**: What, why, how
3. **Reference issues**: `Fixes #123`
4. **Keep commits clean**: Use conventional commits

Example PR description:

```markdown
## Description
Add support for AMD GPU monitoring via rocm-smi.

## Changes
- Add AMD GPUBackend implementation
- Update GPU detection logic
- Add tests for AMD backend

## Testing
- Added unit tests: tests/test_amd_gpu.py
- Tested on AMD Ryzen 9 7950X with RX 7900 XTX

## Related Issues
Fixes #45
```

### Commit Messages

Use conventional commits:

```
feat: add AMD GPU monitoring support
fix: correct VRAM calculation for APUs
docs: update troubleshooting guide
test: add concurrency stress tests
refactor: simplify router engine scoring
```

## Issue Triage

### Bug Reports

When reporting bugs, include:

1. **Environment**: OS, Python version, SmarterRouter version
2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **Logs**: Relevant error messages
5. **Configuration**: Redact sensitive values

### Feature Requests

Include:

1. **Use case**: Why is this needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: What else was considered?
4. **Impact**: Breaking changes? Migration path?

### Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Docs improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## Security

### Reporting Security Issues

**Do not open public issues for security vulnerabilities.**

Email security concerns to: security@smarterrouter.io

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Questions?

- **Discord**: [Join our community](https://discord.gg/smarterrouter)
- **Discussions**: [GitHub Discussions](https://github.com/smarterrouter/smarterrouter/discussions)
- **Email**: dev@smarterrouter.io

Thank you for contributing! 🚀
