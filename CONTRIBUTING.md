# Contributing to SmarterRouter

Thank you for your interest in contributing to SmarterRouter! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SmarterRouter.git
   cd SmarterRouter
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- An LLM backend (Ollama, llama.cpp server, etc.)
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template:
   ```bash
   cp ENV_DEFAULT .env
   # Edit .env with your local configuration
   ```

4. Run the development server:
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436
   ```

## Coding Standards

### Code Style

We use the following tools to maintain code quality:

- **Ruff** for linting and formatting
- **mypy** for type checking

Run these before submitting a PR:

```bash
# Lint and format
ruff check .
ruff format .

# Type check
mypy .
```

### Style Guidelines

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Use absolute imports, sorted alphabetically within groups
- **Types**: Use explicit type annotations, prefer `|` over `Optional` for Python 3.10+
- **Naming**: snake_case for variables/functions, PascalCase for classes, SCREAMING_SNAKE for constants
- **Docstrings**: Use for public functions and classes

### Code Quality Checklist

- [ ] No hardcoded secrets or API keys
- [ ] No `print()` statements in production code (use `logging`)
- [ ] No bare `except:` clauses (catch specific exceptions)
- [ ] All database operations use SQLAlchemy ORM (no raw SQL)
- [ ] All user input is validated and sanitized

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_router.py

# Run specific test
pytest tests/test_router.py::test_select_model_coding_prompt -v
```

### Test Requirements

- All new features must include tests
- All bug fixes should include a regression test
- Maintain or improve code coverage
- Use `pytest.mark.asyncio` for async tests
- Use descriptive test names: `test_<method>_<expected_behavior>`

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestMyFeature:
    """Test class for MyFeature."""
    
    @pytest.fixture
    def mock_dependency(self):
        """Fixture for mocking dependency."""
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_feature_success(self, mock_dependency):
        """Test feature with valid input."""
        # Arrange
        # Act
        # Assert
        pass
```

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the coding standards
3. **Add/update tests** for your changes
4. **Update documentation** if needed (README, AGENTS.md, docstrings)
5. **Run the test suite** and ensure all tests pass
6. **Run linting and type checking**:
   ```bash
   ruff check . && ruff format . && mypy .
   ```
7. **Commit your changes** with a descriptive commit message
8. **Push to your fork** and create a pull request

### PR Checklist

- [ ] Tests pass locally
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] No secrets or sensitive data in commits

### PR Review Process

1. At least one maintainer review is required
2. CI checks must pass
3. Address all review feedback
4. Squash commits before merging (if requested)

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```
feat(router): add support for multimodal routing

Add detection and routing for vision tasks to appropriate models.
Includes tests and documentation updates.

Closes #123
```

```
fix(profiler): prevent timeout crash on slow models

The profiler was crashing when models took longer than the timeout.
Now catches TimeoutError and logs appropriately.

Fixes #456
```

## Questions?

- Open a [Discussion](https://github.com/peva3/SmarterRouter/discussions) for questions
- Open an [Issue](https://github.com/peva3/SmarterRouter/issues) for bugs or feature requests

Thank you for contributing!
