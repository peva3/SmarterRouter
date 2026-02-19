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
    def __init__(self, client: LLMBackend, total_models: int = 0, current_model_num: int = 0, model_name: str = ""):
        self.client = client
        self.judge = JudgeClient()
        self.base_timeout = settings.profile_timeout
        self.timeout = self._calculate_timeout(model_name)
        self.total_models = total_models
        self.current_model_num = current_model_num
        self.start_time = time.perf_counter()
    
    def _calculate_timeout(self, model_name: str) -> float:
        """Calculate appropriate timeout based on model size.
        
        Uses more granular tiers to better handle mid-size models like 7B, 14B, etc.
        """
        params = self._extract_params_from_name(model_name)
        
        if params:
            # More granular timeouts based on model size
            if params >= 70:
                return self.base_timeout * 2.5  # 2.5x for very large (70B+) - 225s
            elif params >= 30:
                return self.base_timeout * 1.8  # 1.8x for large (30-69B) - 162s
            elif params >= 14:
                return self.base_timeout * 1.4  # 1.4x for medium-large (14-29B) - 126s
            elif params >= 7:
                return self.base_timeout * 1.1  # 1.1x for medium (7-13B) - 99s
            elif params <= 3:
                return self.base_timeout * 0.8  # 0.8x for small models (<3B) - 72s
        
        return self.base_timeout  # Default 90s
    
    def _extract_params_from_name(self, model_name: str) -> float | None:
        """Extract parameter count from model name."""
        import re
        
        # Match patterns like "14b", "70B", "8.5b"
        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', model_name)
        if match:
            return float(match.group(1))
        
        # Check for size indicators in name
        name_lower = model_name.lower()
        if any(x in name_lower for x in ['large', 'l', '-l-']):
            return 70.0
        elif any(x in name_lower for x in ['medium', 'm', '-m-']):
            return 14.0
        elif any(x in name_lower for x in ['small', 's', '-s-', 'mini', 'tiny']):
            return 7.0
        
        return None

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
        """Measure current GPU VRAM usage via nvidia-smi (sync version for compatibility)."""
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

    async def _measure_vram_gb_async(self) -> float | None:
        """Measure VRAM asynchronously to avoid blocking the event loop."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=5.0
            )
            
            if proc.returncode == 0:
                used_mb = int(stdout.decode().strip())
                return used_mb / 1024.0
                
        except asyncio.TimeoutError:
            logger.debug("VRAM measurement timed out")
        except Exception as e:
            logger.debug(f"VRAM measurement failed: {e}")
        
        return None

    async def _test_category(
        self,
        model: str,
        category: str,
        prompts: list[str],
    ) -> tuple[float, float]:
        """Process all prompts in a category concurrently with semaphore control."""
        # Limit concurrent prompts to avoid overwhelming the model
        semaphore = asyncio.Semaphore(3)
        
        async def process_single_prompt(prompt: str, prompt_idx: int) -> tuple[float, str, str]:
            """Process a single prompt with semaphore. Returns (elapsed_ms, response_text, prompt)."""
            async with semaphore:
                self._log_progress(model, category, prompt_idx + 1, len(prompts))
                
                try:
                    start = time.perf_counter()
                    result = await asyncio.wait_for(
                        self.client.chat(model, [{"role": "user", "content": prompt}]),
                        timeout=self.timeout,
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    response_text = result.get("message", {}).get("content", "")
                    
                    return elapsed_ms, response_text, prompt
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Profile timeout for {model} on {category} prompt {prompt_idx + 1}")
                    return float(self.timeout * 1000), "", prompt
                except Exception as e:
                    logger.error(f"Profile error for {model}: {e}")
                    return 0.0, "", prompt
        
        # Process all prompts concurrently
        results = await asyncio.gather(*[
            process_single_prompt(p, i) for i, p in enumerate(prompts)
        ])
        
        # Extract times and batch score responses
        times = []
        prompt_response_pairs = []
        empty_indices = []
        
        for i, (elapsed_ms, response_text, prompt) in enumerate(results):
            times.append(elapsed_ms)
            
            if response_text:
                prompt_response_pairs.append((prompt, response_text))
            else:
                empty_indices.append(i)
        
        # Batch score all valid responses (saves API calls)
        if prompt_response_pairs:
            scores = await self.judge.score_responses_batch(prompt_response_pairs, max_concurrent=3)
        else:
            scores = []
        
        # Insert 0.0 for empty responses
        for idx in empty_indices:
            scores.insert(idx, 0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_time = sum(times) / len(times) if times else 0.0
        return avg_score, avg_time

    async def _screen_model(self, model: str) -> tuple[bool, float]:
        """Quick screen with 3 prompts to identify obviously bad models.
        Returns (should_continue, estimated_quality_score).
        """
        # Select one prompt from each category
        screen_prompts = [
            ("reasoning", BENCHMARK_PROMPTS["reasoning"][0]),
            ("coding", BENCHMARK_PROMPTS["coding"][0]),
            ("creativity", BENCHMARK_PROMPTS["creativity"][0]),
        ]
        
        async def test_single(item: tuple[str, str]) -> tuple[float, float]:
            category, prompt = item
            try:
                start = time.perf_counter()
                result = await asyncio.wait_for(
                    self.client.chat(model, [{"role": "user", "content": prompt}]),
                    timeout=self.timeout,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                response_text = result.get("message", {}).get("content", "")
                
                # Quick heuristic: very short responses are likely failures
                if len(response_text.strip()) < 50:
                    return 0.1, elapsed_ms
                return 0.5, elapsed_ms
            except Exception:
                return 0.0, self.timeout * 1000
        
        # Run screening prompts concurrently
        results = await asyncio.gather(*[test_single(item) for item in screen_prompts])
        scores = [r[0] for r in results]
        times = [r[1] for r in results]
        
        avg_score = sum(scores) / len(scores)
        avg_time = sum(times) / len(times)
        
        # If model fails basic screening, skip full profiling
        if avg_score < 0.2 or avg_time > self.timeout * 1000 * 0.9:
            logger.warning(
                f"Model {model} failed screening (score={avg_score:.2f}, time={avg_time:.0f}ms). "
                f"Skipping full profile to save tokens."
            )
            return False, avg_score
        
        return True, avg_score

    async def profile_model(self, model: str) -> ProfileResult | None:
        logger.info(
            f"PROGRESS [{self.current_model_num}/{self.total_models}] Starting profiling: {model}"
        )

        # Phase 1: Quick screening (saves judge tokens on bad models)
        should_continue, screen_score = await self._screen_model(model)
        
        if not should_continue:
            # Return minimal profile for failed models
            result = ProfileResult(
                model_name=model,
                reasoning=screen_score * 0.3,
                coding=screen_score * 0.3,
                creativity=screen_score * 0.3,
                speed=0.3,
                avg_response_time_ms=self.timeout * 1000,
                vision=self._detect_vision_capability(model),
                tool_calling=self._detect_tool_capability(model),
            )
            self._save_profile(result, vram_gb=None)
            return result

        # VRAM measurement: Use Ollama API if available, fallback to nvidia-smi delta
        measured_vram_gb: float | None = None
        
        # Run all categories concurrently for speed
        categories = ["reasoning", "coding", "creativity"]
        category_tasks = [
            self._test_category(model, cat, BENCHMARK_PROMPTS[cat])
            for cat in categories
        ]
        
        results = await asyncio.gather(*category_tasks, return_exceptions=True)
        
        # Unpack results with error handling
        category_scores = {}
        category_times = {}
        
        for i, (cat, result) in enumerate(zip(categories, results)):
            if isinstance(result, Exception):
                logger.error(f"Category {cat} failed for {model}: {result}")
                category_scores[cat] = 0.0
                category_times[cat] = self.timeout * 1000
            else:
                score, time_ms = result
                category_scores[cat] = score
                category_times[cat] = time_ms
        
        # Early termination check: if first category (reasoning) is terrible, note it
        if category_scores.get("reasoning", 0) < 0.1:
            logger.warning(
                f"Model {model} has very poor reasoning score ({category_scores['reasoning']:.2f})"
            )

        # VRAM measurement: Try Ollama API first, then fallback to nvidia-smi
        if settings.profile_measure_vram:
            # Method 1: Query Ollama directly for per-model VRAM (most accurate)
            try:
                ollama_vram = await self.client.get_model_vram_usage(model)
                if ollama_vram is not None and ollama_vram > 0:
                    measured_vram_gb = ollama_vram
                    logger.info(f"VRAM measured via Ollama API for {model}: {measured_vram_gb:.2f}GB")
            except Exception as e:
                logger.debug(f"Ollama VRAM query failed for {model}: {e}")
            
            # Method 2: Fallback to nvidia-smi delta measurement
            if measured_vram_gb is None:
                baseline_vram_gb = await self._measure_vram_gb_async()
                if baseline_vram_gb is not None:
                    logger.debug(f"VRAM baseline for {model}: {baseline_vram_gb:.2f}GB")
                    # Small delay to let VRAM stabilize
                    await asyncio.sleep(0.5)
                    after_vram_gb = await self._measure_vram_gb_async()
                    if after_vram_gb is not None:
                        delta = after_vram_gb - baseline_vram_gb
                        if delta > 0.1:  # Only accept if delta is meaningful (>100MB)
                            measured_vram_gb = delta
                            logger.info(f"VRAM measured via nvidia-smi for {model}: {measured_vram_gb:.2f}GB")
                        else:
                            logger.debug(f"VRAM delta too small for {model}: {delta:.2f}GB (model may have unloaded)")

        all_times = list(category_times.values())
        avg_time = sum(all_times) / len(all_times)
        speed_score = 1.0 - min(avg_time / 30000.0, 1.0)

        # Capability Detection
        vision_capable = self._detect_vision_capability(model)
        tool_capable = self._detect_tool_capability(model)

        result = ProfileResult(
            model_name=model,
            reasoning=category_scores["reasoning"],
            coding=category_scores["coding"],
            creativity=category_scores["creativity"],
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
            profiler = ModelProfiler(client, total_models=len(new_models), current_model_num=i, model_name=model_info.name)
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
