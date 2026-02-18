import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import re

from router.backends.base import LLMBackend
from router.config import settings
from router.database import get_session
from router.judge import JudgeClient
from router.models import ModelProfile
from router.prompts import BENCHMARK_PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    model_name: str
    reasoning: float
    coding: float
    creativity: float
    speed: float
    avg_response_time_ms: float
    vision: bool = False
    tool_calling: bool = False


class ModelProfiler:
    def __init__(self, client: LLMBackend, total_models: int = 0, current_model_num: int = 0):
        self.client = client
        self.judge = JudgeClient()
        self.timeout = settings.profile_timeout
        self.total_models = total_models
        self.current_model_num = current_model_num
        self.start_time = time.perf_counter()

    def _log_progress(self, model: str, category: str, prompt_num: int, total_prompts: int) -> None:
        """Log current profiling progress with ETA."""
        elapsed = time.perf_counter() - self.start_time

        # Calculate progress
        cat_list = list(BENCHMARK_PROMPTS.keys())
        try:
            categories_completed = cat_list.index(category)
        except ValueError:
            categories_completed = 0

        prompts_per_cat = len(BENCHMARK_PROMPTS.get(category, [1]))
        prompts_completed = categories_completed * prompts_per_cat + prompt_num
        total_prompts_all = sum(len(p) for p in BENCHMARK_PROMPTS.values())

        if self.total_models > 0:
            models_completed = self.current_model_num - 1 + (prompts_completed / total_prompts_all)
            eta_total = (
                (elapsed / models_completed) * self.total_models if models_completed > 0 else 0
            )
            eta_str = f"ETA: {int(eta_total / 60)}m {int(eta_total % 60)}s remaining"
        else:
            eta_str = "Calculating ETA..."

        progress_pct = (prompts_completed / total_prompts_all) * 100

        logger.info(
            f"PROGRESS [{self.current_model_num}/{self.total_models}] {model} - "
            f"{category} ({prompt_num}/{total_prompts}) - "
            f"{progress_pct:.1f}% - {eta_str}"
        )

    def _measure_vram_gb(self) -> float | None:
        """Measure current GPU VRAM usage via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                used_mb = int(result.stdout.strip())
                return used_mb / 1024.0
        except Exception as e:
            logger.debug(f"VRAM measurement failed: {e}")
        return None

    async def _test_category(
        self,
        model: str,
        category: str,
        prompts: list[str],
    ) -> tuple[float, float]:
        scores: list[float] = []
        times: list[float] = []

        for i, prompt in enumerate(prompts, 1):
            self._log_progress(model, category, i, len(prompts))

            try:
                start = time.perf_counter()
                result = await asyncio.wait_for(
                    self.client.chat(model, [{"role": "user", "content": prompt}]),
                    timeout=self.timeout,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                # Extract response text from chat format (normalized to Ollama format)
                response_text = result.get("message", {}).get("content", "")

                # Use Judge if enabled
                score = await self.judge.score_response(prompt, response_text)
                scores.append(score)

                times.append(elapsed_ms)

            except asyncio.TimeoutError:
                logger.warning(f"Profile timeout for {model} on category {category}")
                scores.append(0.0)
                times.append(self.timeout * 1000)
            except Exception as e:
                logger.error(f"Profile error for {model}: {e}")
                scores.append(0.0)
                times.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_time = sum(times) / len(times) if times else 0.0
        return avg_score, avg_time

    async def profile_model(self, model: str) -> ProfileResult | None:
        logger.info(
            f"PROGRESS [{self.current_model_num}/{self.total_models}] Starting profiling: {model}"
        )

        # VRAM measurement: baseline before any generation
        baseline_vram_gb: float | None = None
        measured_vram_gb: float | None = None
        if settings.profile_measure_vram:
            baseline_vram_gb = self._measure_vram_gb()
            if baseline_vram_gb is not None:
                logger.debug(
                    f"VRAM baseline for {model}: {baseline_vram_gb:.2f}GB (before generation)"
                )

        reasoning_score, reasoning_time = await self._test_category(
            model, "reasoning", BENCHMARK_PROMPTS["reasoning"]
        )
        coding_score, coding_time = await self._test_category(
            model, "coding", BENCHMARK_PROMPTS["coding"]
        )
        creativity_score, creativity_time = await self._test_category(
            model, "creativity", BENCHMARK_PROMPTS["creativity"]
        )

        # VRAM measurement: after all categories to get model's total footprint
        if settings.profile_measure_vram and baseline_vram_gb is not None:
            after_vram_gb = self._measure_vram_gb()
            if after_vram_gb is not None:
                delta = after_vram_gb - baseline_vram_gb
                # Only accept positive deltas (model should increase VRAM usage)
                if delta > 0:
                    measured_vram_gb = delta
                    logger.info(f"VRAM measured for {model}: {measured_vram_gb:.2f}GB")
                else:
                    logger.warning(
                        f"VRAM measurement invalid for {model}: after={after_vram_gb:.2f}, baseline={baseline_vram_gb:.2f}"
                    )

        all_times = [reasoning_time, coding_time, creativity_time]
        avg_time = sum(all_times) / len(all_times)

        speed_score = 1.0 - min(avg_time / 30000.0, 1.0)

        # Capability Detection
        vision_capable = self._detect_vision_capability(model)
        tool_capable = self._detect_tool_capability(model)

        result = ProfileResult(
            model_name=model,
            reasoning=reasoning_score,
            coding=coding_score,
            creativity=creativity_score,
            speed=speed_score,
            avg_response_time_ms=avg_time,
            vision=vision_capable,
            tool_calling=tool_capable,
        )

        self._save_profile(result, vram_gb=measured_vram_gb)

        total_score = result.reasoning + result.coding + result.creativity + result.speed
        elapsed_total = time.perf_counter() - self.start_time

        logger.info(
            f"PROGRESS [{self.current_model_num}/{self.total_models}] Profile complete for {model}: "
            f"reasoning={result.reasoning:.2f}, coding={result.coding:.2f}, "
            f"creativity={result.creativity:.2f}, "
            f"speed={result.speed:.2f}, vision={result.vision}, tools={result.tool_calling}, "
            f"total={total_score:.2f}, time={elapsed_total:.1f}s"
            + (f", vram={measured_vram_gb:.2f}GB" if measured_vram_gb else "")
        )

        return result

    def _detect_vision_capability(self, model_name: str) -> bool:
        """Heuristic detection of vision capabilities."""
        name = model_name.lower()
        vision_keywords = [
            "vision",
            "llava",
            "pixtral",
            "gpt-4o",
            "gemini",
            "claude-3",
            "minicpm",
            "moondream",
            "vl",
        ]
        return any(kw in name for kw in vision_keywords)

    def _detect_tool_capability(self, model_name: str) -> bool:
        """Heuristic detection of tool calling capabilities."""
        name = model_name.lower()
        tool_keywords = [
            "gpt-4",
            "claude-3",
            "mistral-large",
            "qwen2.5",
            "llama3.1",
            "command-r",
            "hermes",
            "tool",
        ]
        return any(kw in name for kw in tool_keywords)

    def _save_profile(self, result: ProfileResult, vram_gb: float | None = None) -> None:
        try:
            with get_session() as session:
                profile = session.query(ModelProfile).filter_by(name=result.model_name).first()

                if profile:
                    profile.reasoning = result.reasoning
                    profile.coding = result.coding
                    profile.creativity = result.creativity
                    profile.speed = result.speed
                    profile.avg_response_time_ms = result.avg_response_time_ms
                    profile.last_profiled = datetime.now(timezone.utc)
                    profile.vision = result.vision
                    profile.tool_calling = result.tool_calling
                    # Update VRAM fields if measured
                    if vram_gb is not None:
                        profile.vram_required_gb = vram_gb
                        profile.vram_measured_at = datetime.now(timezone.utc)
                else:
                    profile = ModelProfile(
                        name=result.model_name,
                        reasoning=result.reasoning,
                        coding=result.coding,
                        creativity=result.creativity,
                        speed=result.speed,
                        avg_response_time_ms=result.avg_response_time_ms,
                        last_profiled=datetime.now(timezone.utc),
                        vision=result.vision,
                        tool_calling=result.tool_calling,
                        vram_required_gb=vram_gb,
                        vram_measured_at=datetime.now(timezone.utc)
                        if vram_gb is not None
                        else None,
                    )
                session.add(profile)
                session.commit()
                logger.debug(
                    f"Saved profile for {result.model_name}"
                    + (f" with VRAM {vram_gb:.2f}GB" if vram_gb else "")
                )
        except Exception as e:
            logger.error(f"Failed to save profile for {result.model_name}: {e}")


async def profile_all_models(client: LLMBackend, force: bool = False) -> list[ProfileResult]:
    models = await client.list_models()
    total_models = len(models)

    existing_profiles = get_all_profile_names()
    ollama_model_names = {m.name for m in models}
    new_models = [m for m in models if m.name not in existing_profiles]

    if not force and existing_profiles:
        logger.info(
            f"Found {len(existing_profiles)} existing profiles, profiling {len(new_models)} new models"
        )
    else:
        new_models = models
        logger.info(f"PROGRESS: Starting profiling of {total_models} models (force={force})")

    if not new_models:
        logger.info("All models already profiled")
        return []

    results: list[ProfileResult] = []
    for i, model_info in enumerate(new_models, 1):
        try:
            profiler = ModelProfiler(client, total_models=len(new_models), current_model_num=i)
            result = await profiler.profile_model(model_info.name)
            if result:
                results.append(result)
                logger.info(
                    f"PROGRESS: Completed {i}/{len(new_models)} models ({(i / len(new_models)) * 100:.1f}%)"
                )
        except Exception as e:
            logger.error(f"Failed to profile {model_info.name}: {e}")

    logger.info(
        f"PROGRESS: Profiling complete! {len(results)}/{len(new_models)} models profiled successfully"
    )
    return results


def get_all_profile_names() -> set[str]:
    with get_session() as session:
        profiles = session.query(ModelProfile.name).all()
        return {p.name for p in profiles}
