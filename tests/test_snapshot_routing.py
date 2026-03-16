"""Snapshot testing for routing heuristics (Item #57).

These tests snapshot deterministic prompt-analysis output from RouterEngine and
guard against accidental behavior drift.
"""

import json
import os
from pathlib import Path

import pytest

from router.backends.base import ModelInfo
from router.router import RouterEngine

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


class _DummyBackend:
    """Minimal backend to satisfy RouterEngine construction in tests."""

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="llama3:8b"), ModelInfo(name="mistral:7b")]

    async def chat(self, *args, **kwargs) -> dict:
        return {"message": {"content": "ok"}}

    async def chat_streaming(self, *args, **kwargs):
        raise NotImplementedError

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def load_model(self, model_name: str, keep_alive: float = -1, timeout: float | None = None) -> bool:
        return False

    async def embed(self, model: str, input_text, **kwargs) -> dict:
        return {"embedding": [0.1, 0.2, 0.3]}

    async def get_model_vram_usage(self, model_name: str) -> float | None:
        return None

    async def close(self) -> None:
        return None

    def is_external_model(self, model_name: str) -> bool:
        return False


def _load_snapshot(name: str) -> dict | None:
    path = SNAPSHOTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_snapshot(name: str, data: dict) -> None:
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    path = SNAPSHOTS_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


@pytest.fixture
def engine() -> RouterEngine:
    return RouterEngine(client=_DummyBackend(), cache_enabled=False)


@pytest.fixture
def prompts() -> list[str]:
    return [
        "Write a Python function to calculate fibonacci",
        "Explain quantum computing in simple terms",
        "Create a poem about artificial intelligence",
        "What is the capital of France?",
        "Debug this SQL query and optimize it",
    ]


def test_update_prompt_analysis_snapshot(engine: RouterEngine, prompts: list[str]) -> None:
    if os.getenv("UPDATE_SNAPSHOTS") != "1":
        pytest.skip("Set UPDATE_SNAPSHOTS=1 to update snapshots")

    data = {prompt: engine._analyze_prompt(prompt) for prompt in prompts}
    _save_snapshot("prompt_analysis", data)


def test_prompt_analysis_matches_snapshot(engine: RouterEngine, prompts: list[str]) -> None:
    snapshot = _load_snapshot("prompt_analysis")
    if snapshot is None:
        pytest.skip("No snapshot found. Run with UPDATE_SNAPSHOTS=1 to create.")

    mismatches: list[str] = []
    for prompt in prompts:
        expected = snapshot.get(prompt)
        actual = engine._analyze_prompt(prompt)
        if expected != actual:
            mismatches.append(f"{prompt!r}: expected={expected}, actual={actual}")

    if mismatches:
        pytest.fail("Prompt analysis snapshot mismatch:\n" + "\n".join(mismatches))
