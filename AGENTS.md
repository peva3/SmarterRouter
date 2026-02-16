# AGENTS.md - LLM Router Proxy

## Project Overview

AI-powered LLM router that sits between OpenWebUI and various LLM backends. Discovers available models, profiles their capabilities via test prompts, and uses AI to intelligently route queries to the best model. Features live model detection, response signatures, and support for multiple backends (Ollama, llama.cpp, OpenAI-compatible APIs).

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **LLM Backend**: httpx (async HTTP) - abstracted via backend interface
- **Storage**: SQLite with SQLAlchemy
- **Testing**: pytest + pytest-asyncio
- **Deployment**: Docker + Docker Compose

---

## Commands

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (development)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436
```

### Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=term-missing

# Run a single test file
pytest tests/test_router.py

# Run a single test function
pytest tests/test_router.py::test_select_best_model

# Run tests matching pattern
pytest -k "test_profile"

# Run with verbose output
pytest -v -s

# Run async tests
pytest tests/ --asyncio-mode=auto
```

### Linting & Type Checking

```bash
# Lint with ruff
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format with ruff
ruff format .

# Type check with mypy
mypy .

# Run all checks
ruff check . && ruff format . && mypy .
```

### Docker

```bash
# Build image
docker build -t llm-router .

# Run container
docker run -p 11434:11434 --env-file .env llm-router

# Docker Compose (from project root)
docker-compose up -d --build
```

---

## Code Style

### General

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Newlines**: LF endings, trailing newline at EOF
- **Encoding**: UTF-8

### Imports

```python
# Standard library first, then third-party, then local
import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx
from fastapi import FastAPI, Request
from sqlalchemy import select

from router.config import Settings
from router.models import ModelProfile
from router.profiler import ModelProfiler
```

- Use absolute imports (not relative `..`)
- Sort imports alphabetically within groups
- Use `import x` not `from x import y` unless avoiding conflicts
- Group with blank lines between sections

### Types

```python
# Prefer explicit types, especially for function signatures
def route_query(prompt: str, models: list[ModelProfile]) -> str:
    """Route a prompt to the best model."""
    selected: str | None = None
    scores: dict[str, float] = {}
    return selected

# Use | instead of Optional for Python 3.10+
user_id: int | None = None

# Use built-in collections (list, dict) not typing classes
def process(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    return counts
```

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Variables | snake_case | `model_name`, `response_text` |
| Functions | snake_case | `get_best_model()`, `profile_model()` |
| Classes | PascalCase | `ModelProfiler`, `RouterEngine` |
| Constants | SCREAMING_SNAKE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private methods | _snake_case | `_fetch_models()`, `_calculate_score()` |
| Config fields | snake_case | `ollama_url`, `signature_enabled` |

### Error Handling

```python
# Use custom exceptions for domain errors
class ProfilingError(Exception):
    """Raised when model profiling fails."""
    pass


class RouterError(Exception):
    """Raised when routing decision fails."""
    pass


# Handle gracefully with context
async def call_ollama(prompt: str, model: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/generate",
                json={"model": model, "prompt": prompt}
            )
            response.raise_for_status()
            return response.json()["response"]
    except httpx.TimeoutException:
        logger.warning(f"Timeout calling model {model}")
        raise RouterError(f"Model {model} timed out") from None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling {model}: {e}")
        raise RouterError(f"Model {model} unavailable") from None
```

- Always include context in exceptions (use `from None` to suppress chain)
- Log at appropriate level (DEBUG for retryable, ERROR for fatal)
- Use `None` return with type annotation, not tuple

### Async

```python
# Use async/await consistently
async def fetch_models() -> list[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        return [m["name"] for m in response.json()["models"]]

# Use asyncio.gather for parallel operations
async def profile_models(models: list[str]) -> dict[str, Profile]:
    tasks = [profile_model(m) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {m: r for m, r in zip(models, results) if not isinstance(r, Exception)}
```

### Configuration

```python
# Use Pydantic Settings for configuration
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Provider selection
    provider: str = "ollama"  # ollama, llama.cpp, openai
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    pinned_model: str | None = None  # Model to keep in VRAM
    
    # llama.cpp settings
    llama_cpp_url: str | None = None
    
    # OpenAI-compatible settings
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    model_prefix: str = ""
    
    # Other settings
    signature_enabled: bool = True
    signature_format: str = "\nModel: {model}"
    polling_interval: int = 60
    profile_timeout: int = 30
    generation_timeout: int = 120  # Timeout for model generation

    class Config:
        env_file = ".env"
        env_prefix = "ROUTER_"
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)


def init_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
```

### Database (SQLAlchemy)

```python
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ModelProfile(Base):
    __tablename__ = "model_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    reasoning = Column(Float, default=0.0)
    coding = Column(Float, default=0.0)
    creativity = Column(Float, default=0.0)
    last_profiled = Column(DateTime)
```

### Testing

```python
import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_ollama():
    """Fixture for mocking Ollama responses."""
    return AsyncMock()


@pytest.mark.asyncio
async def test_select_best_model():
    """Test model selection logic."""
    profiles = [
        ModelProfile(name="llama3", reasoning=0.9, coding=0.7),
        ModelProfile(name="codellama", reasoning=0.7, coding=0.95),
    ]
    prompt = "Write a Python function to calculate fibonacci"

    result = await route_query(prompt, profiles)

    assert result == "codellama"


# Use pytest.mark.asyncio for all async tests
# Use descriptive test names: test_<method>_<expected_behavior>
# Assert meaningfully, not just truthy/falsy values
```

---

## Project Structure

```
.
├── main.py              # FastAPI app, endpoints
├── router/
│   ├── __init__.py
│   ├── config.py        # Settings
│   ├── models.py        # Pydantic/SQLAlchemy models
│   ├── router.py        # Routing logic
│   ├── profiler.py      # Model profiling
│   └── backends/        # Backend implementations
│       ├── __init__.py  # Factory function
│       ├── base.py      # LLMBackend Protocol
│       ├── ollama.py    # Ollama backend
│       ├── llama_cpp.py # llama.cpp backend
│       └── openai.py    # OpenAI-compatible backend
├── tests/
│   ├── test_router.py
│   └── test_profiler.py
├── .env                 # Environment variables
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── AGENTS.md
```

---

## Cursor/Copilot Rules

No existing Cursor or Copilot rules found in this repository.

---

## Notes

- All async HTTP calls use httpx (not requests)
- Configuration via environment variables with `ROUTER_` prefix
- SQLite database stored in `router.db`
- Default port matches Ollama (11434) for drop-in replacement
- Signature appended to every response unless disabled
- Supports multiple backends: Ollama, llama.cpp server, OpenAI-compatible APIs
- Backend abstraction via LLMBackend Protocol in `router/backends/`
- VRAM management: Set `pinned_model` to keep a small model in VRAM
- Generation timeout: Set `generation_timeout` for large models (default 120s)
- Backward compatible: Default `provider=ollama` preserves existing behavior
