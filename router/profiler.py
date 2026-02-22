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
    def __init__(
        self,
        client: LLMBackend,
        total_models: int = 0,
        current_model_num: int = 0,
        model_name: str = "",
        model_size_bytes: int | None = None,
    ):
        self.client = client
        self.judge = JudgeClient()
        self.base_timeout = settings.profile_timeout
        self.timeout = self._calculate_timeout(model_name)
        self.model_size_bytes = model_size_bytes
        self.total_models = total_models
        self.current_model_num = current_model_num
        self.start_time = time.perf_counter()
        # Adaptive timeout tracking
        self.adaptive_timeout: float | None = None
        self.screening_token_rate: float | None = None
        self.screening_max_time_ms: float = 0.0
        self.screening_total_tokens: int = 0
    
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

    def _calculate_warmup_timeout(self) -> float:
        """Calculate a generous timeout for model warmup/loading based on disk size.
        
        Uses configurable disk read speed assumption to ensure even slow HDDs
        have enough time to load large models.
        
        Formula: (size_gb / disk_speed_gbps) + 30s buffer
        - 1.5GB model: ~60 seconds
        - 14GB model: ~5 minutes
        - 40GB model: ~13+ minutes
        """
        if self.model_size_bytes is None:
            # Fallback to generous default if size unknown
            return 600.0  # 10 minutes
        
        size_gb = self.model_size_bytes / (1024 ** 3)
        # Get disk speed config (convert MB/s to GB/s)
        disk_speed_mbps = settings.profile_warmup_disk_speed_mbps
        disk_speed_gbps = disk_speed_mbps / 1024.0
        
        # Calculate load time
        load_seconds = size_gb / disk_speed_gbps if disk_speed_gbps > 0 else float('inf')
        buffer_seconds = 30.0
        warmup_timeout = load_seconds + buffer_seconds
        
        # Cap at configured maximum to avoid infinite waits for truly broken models
        return min(warmup_timeout, settings.profile_warmup_max_timeout)

    async def _warmup_model(self, model: str) -> bool:
        """Explicitly load the model into memory before benchmarking.
        
        This separates the slow I/O-bound loading phase from the compute-bound
        benchmarking phase, preventing timeouts caused by slow disk reads.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        warmup_timeout = self._calculate_warmup_timeout()
        size_str = f"{self.model_size_bytes/(1024**3):.1f}GB" if self.model_size_bytes else "unknown"
        logger.info(
            f"Warming up model {model} (size={size_str}, timeout={warmup_timeout:.0f}s)"
        )
        
        try:
            success = await asyncio.wait_for(
                self.client.load_model(model, keep_alive=-1),
                timeout=warmup_timeout,
            )
            if success:
                logger.info(f"Model {model} warmed up successfully")
            else:
                logger.warning(f"Model {model} warmup returned False (backend may not support explicit loading)")
            return success
        except asyncio.TimeoutError:
            logger.error(f"Model {model} warmup timed out after {warmup_timeout:.0f}s")
            return False
        except Exception as e:
            logger.error(f"Model {model} warmup failed: {e}")
            return False

    def _calculate_adaptive_timeout(
        self,
        screening_total_time_ms: float,
        screening_max_time_ms: float,
        screening_total_tokens: int,
        model_name: str = "",
    ) -> float:
        """Calculate adaptive timeout based on measured screening performance.
        
        Uses two methods and takes the maximum:
        1. Method A (Conservative): max_prompt_time × safety_factor
        2. Method B (Token-based): projected total time based on token rate
        
        Args:
            screening_total_time_ms: Total time for all screening prompts
            screening_max_time_ms: Longest single screening prompt time
            screening_total_tokens: Total tokens generated during screening
            model_name: Name of the model to check for reasoning capabilities
            
        Returns:
            Calculated timeout in seconds, bounded by min/max config
        """
        prompts_per_category = settings.profile_prompts_per_category
        total_benchmark_prompts = 3 * prompts_per_category
        screening_prompts = 3
        
        # Determine safety factor
        safety_factor = settings.profile_adaptive_safety_factor
        
        # Reasoning models (like DeepSeek R1) are non-linear; they take MUCH longer
        # on complex benchmarks than simple screening prompts.
        name_lower = model_name.lower()
        if any(x in name_lower for x in ["r1", "reasoning", "thought", "cot"]):
            safety_factor *= 2.0  # Double safety for reasoning models (e.g. 2.0 -> 4.0)
            logger.debug(f"Detected reasoning model {model_name}, bumping safety factor to {safety_factor}")

        # Method A: Conservative - use max prompt time × safety factor
        timeout_a = (screening_max_time_ms / 1000.0) * safety_factor
        
        # Method B: Token-based projection
        timeout_b = timeout_a  # fallback to A
        if screening_total_time_ms > 0 and screening_total_tokens > 0:
            token_rate = screening_total_tokens / (screening_total_time_ms / 1000.0)
            self.screening_token_rate = token_rate
            
            # Project remaining tokens needed
            avg_tokens_per_prompt = screening_total_tokens / screening_prompts
            remaining_prompts = total_benchmark_prompts - screening_prompts
            remaining_tokens = avg_tokens_per_prompt * remaining_prompts
            
            # Project total time including screening
            projected_remaining_time = remaining_tokens / token_rate if token_rate > 0 else 0
            projected_total_time = (screening_total_time_ms / 1000.0) + projected_remaining_time
            
            # Per-prompt timeout with safety factor
            timeout_b = (projected_total_time / total_benchmark_prompts) * settings.profile_adaptive_safety_factor
        
        # Use the more conservative (higher) timeout
        calculated_timeout = max(timeout_a, timeout_b)
        
        # PRO-TIP: Use the size-based timeout as a FLOOR.
        # We should never give a model LESS time than our size-based guess,
        # only MORE if the screening shows it's particularly slow.
        floor_timeout = self.timeout
        
        # Apply min/max and floor
        final_timeout = max(
            floor_timeout,
            settings.profile_adaptive_timeout_min,
            min(calculated_timeout, settings.profile_adaptive_timeout_max)
        )
        
        logger.info(
            f"Adaptive timeout calculated: {final_timeout:.0f}s "
            f"(floor={floor_timeout:.0f}s, method_a={timeout_a:.0f}s, method_b={timeout_b:.0f}s, "
            f"token_rate={self.screening_token_rate:.1f} tok/s)"
        )
        
        return final_timeout

    async def _test_category(
        self,
        model: str,
        category: str,
        prompts: list[str],
        timeout_override: float | None = None,
    ) -> tuple[float, float]:
        """Process all prompts in a category concurrently with semaphore control."""
        effective_timeout = timeout_override if timeout_override is not None else self.timeout
        
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
                        timeout=effective_timeout,
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    response_text = result.get("message", {}).get("content", "")
                    
                    return elapsed_ms, response_text, prompt
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Profile timeout for {model} on {category} prompt {prompt_idx + 1}")
                    return float(effective_timeout * 1000), "", prompt
                except Exception as e:
                    logger.error(f"Profile error for {model}: {type(e).__name__}: {e}")
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

    async def _screen_model(self, model: str) -> tuple[bool, float, float, float, int]:
        """Quick screen with 3 prompts to identify obviously bad models.
        
        Returns:
            tuple of (should_continue, avg_score, total_time_ms, max_time_ms, total_tokens)
        """
        # Select one prompt from each category
        screen_prompts = [
            ("reasoning", BENCHMARK_PROMPTS["reasoning"][0]),
            ("coding", BENCHMARK_PROMPTS["coding"][0]),
            ("creativity", BENCHMARK_PROMPTS["creativity"][0]),
        ]
        
        async def test_single(item: tuple[str, str]) -> tuple[float, float, int]:
            category, prompt = item
            try:
                start = time.perf_counter()
                result = await asyncio.wait_for(
                    self.client.chat(model, [{"role": "user", "content": prompt}]),
                    timeout=self.timeout,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                response_text = result.get("message", {}).get("content", "")
                # Get token count from response if available
                token_count = result.get("eval_count", 0)
                # Fallback: estimate from response length
                if token_count == 0 and response_text:
                    token_count = int(len(response_text.split()) * 1.3)
                
                # Quick heuristic: very short responses are likely failures
                if len(response_text.strip()) < 50:
                    return 0.1, elapsed_ms, token_count
                return 0.5, elapsed_ms, token_count
            except Exception:
                return 0.0, self.timeout * 1000, 0
        
        # Run screening prompts concurrently
        results = await asyncio.gather(*[test_single(item) for item in screen_prompts])
        scores = [r[0] for r in results]
        times = [r[1] for r in results]
        tokens = [r[2] for r in results]
        
        avg_score = sum(scores) / len(scores)
        total_time = sum(times)
        max_time = max(times)
        total_tokens = sum(tokens)
        
        # If model fails basic screening, skip full profiling
        if avg_score < 0.2 or max_time > self.timeout * 1000 * 0.95:
            logger.warning(
                f"Model {model} failed screening (score={avg_score:.2f}, max_time={max_time:.0f}ms). "
                f"Skipping full profile to save tokens."
            )
            return False, avg_score, total_time, max_time, total_tokens
        
        return True, avg_score, total_time, max_time, total_tokens

    async def profile_model(self, model: str) -> ProfileResult | None:
        logger.info(
            f"PROGRESS [{self.current_model_num}/{self.total_models}] Starting profiling: {model}"
        )

        # Phase 0: Warm up the model (load into memory with generous timeout)
        # This prevents timeouts due to slow disk I/O during actual benchmarking
        warmup_success = await self._warmup_model(model)
        if not warmup_success:
            logger.warning(
                f"Model {model} warmup failed. Profiling may be inaccurate or time out. "
                "Continuing anyway to attempt profiling..."
            )
            # Don't completely fail - some backends may not support explicit loading
            # or the model might already be loaded. Continue to screening.

        # Phase 1: Quick screening (saves judge tokens on bad models)
        # Also captures performance metrics for adaptive timeout calculation
        should_continue, screen_score, screen_total_ms, screen_max_ms, screen_tokens = await self._screen_model(model)
        
        # Store screening metrics
        self.screening_max_time_ms = screen_max_ms
        self.screening_total_tokens = screen_tokens
        
        if not should_continue:
            # Return minimal profile for failed models
            result = ProfileResult(
                model_name=model,
                reasoning=screen_score * 0.3,
                coding=screen_score * 0.3,
                creativity=screen_score * 0.3,
                speed=0.3,
                avg_response_time_ms=screen_max_ms,
                vision=self._detect_vision_capability(model),
                tool_calling=self._detect_tool_capability(model),
            )
            self._save_profile(result, vram_gb=None)
            return result

        # Phase 2: Calculate adaptive timeout based on screening performance
        self.adaptive_timeout = self._calculate_adaptive_timeout(
            screening_total_time_ms=screen_total_ms,
            screening_max_time_ms=screen_max_ms,
            screening_total_tokens=screen_tokens,
            model_name=model,
        )
        logger.info(f"Using adaptive timeout of {self.adaptive_timeout:.0f}s for {model}")

        # VRAM measurement: Use Ollama API if available, fallback to nvidia-smi delta
        measured_vram_gb: float | None = None
        
        # Run all categories concurrently for speed with adaptive timeout
        categories = ["reasoning", "coding", "creativity"]
        category_tasks = [
            self._test_category(model, cat, BENCHMARK_PROMPTS[cat], timeout_override=self.adaptive_timeout)
            for cat in categories
        ]
        
        results = await asyncio.gather(*category_tasks, return_exceptions=True)
        
        # Unpack results with error handling
        category_scores: dict[str, float] = {}
        category_times: dict[str, float] = {}
        
        for i, cat in enumerate(categories):
            cat_result = results[i]
            if isinstance(cat_result, Exception):
                logger.error(f"Category {cat} failed for {model}: {cat_result}")
                category_scores[cat] = 0.0
                category_times[cat] = self.timeout * 1000
            else:
                # At this point, result should be tuple[float, float]
                try:
                    score, time_ms = cat_result  # type: ignore[misc]
                    category_scores[cat] = float(score)
                    category_times[cat] = float(time_ms)
                except (TypeError, ValueError) as e:
                    logger.error(f"Unexpected result format for {cat}: {cat_result}, error: {e}")
                    category_scores[cat] = 0.0
                    category_times[cat] = self.timeout * 1000
        
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

        self._save_profile(
            result,
            vram_gb=measured_vram_gb,
            adaptive_timeout=self.adaptive_timeout,
            token_rate=self.screening_token_rate,
        )

        total_score = result.reasoning + result.coding + result.creativity + result.speed
        elapsed_total = time.perf_counter() - self.start_time

        logger.info(
            f"PROGRESS [{self.current_model_num}/{self.total_models}] Profile complete for {model}: "
            f"reasoning={result.reasoning:.2f}, coding={result.coding:.2f}, "
            f"creativity={result.creativity:.2f}, "
            f"speed={result.speed:.2f}, vision={result.vision}, tools={result.tool_calling}, "
            f"total={total_score:.2f}, time={elapsed_total:.1f}s"
            + (f", vram={measured_vram_gb:.2f}GB" if measured_vram_gb else "")
            + (f", timeout={self.adaptive_timeout:.0f}s" if self.adaptive_timeout else "")
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

    def _save_profile(
        self,
        result: ProfileResult,
        vram_gb: float | None = None,
        adaptive_timeout: float | None = None,
        token_rate: float | None = None,
    ) -> None:
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
                    # Update adaptive timeout metrics
                    if adaptive_timeout is not None:
                        profile.adaptive_timeout_used = adaptive_timeout
                    if token_rate is not None:
                        profile.profiling_token_rate = token_rate
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
                        adaptive_timeout_used=adaptive_timeout,
                        profiling_token_rate=token_rate,
                    )
                session.add(profile)
                session.commit()
                logger.debug(
                    f"Saved profile for {result.model_name}"
                    + (f" with VRAM {vram_gb:.2f}GB" if vram_gb else "")
                    + (f", timeout={adaptive_timeout:.0f}s" if adaptive_timeout else "")
                )
        except Exception as e:
            logger.error(f"Failed to save profile for {result.model_name}: {e}")


async def profile_all_models(client: LLMBackend, force: bool = False) -> list[ProfileResult]:
    """Profile all models, optionally in parallel based on configuration."""
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
    parallel_count = max(1, settings.profile_parallel_count)
    
    if parallel_count > 1:
        logger.info(f"Profiling {len(new_models)} models with parallelism={parallel_count}")
        
        semaphore = asyncio.Semaphore(parallel_count)
        completed_count = 0
        lock = asyncio.Lock()
        
        async def profile_single(model_info: ModelInfo, index: int) -> ProfileResult | None:
            nonlocal completed_count
            async with semaphore:
                try:
                    profiler = ModelProfiler(
                        client,
                        total_models=len(new_models),
                        current_model_num=index,
                        model_name=model_info.name,
                        model_size_bytes=model_info.size,
                    )
                    result = await profiler.profile_model(model_info.name)
                    
                    async with lock:
                        completed_count += 1
                        logger.info(
                            f"PROGRESS: Completed {completed_count}/{len(new_models)} models "
                            f"({(completed_count / len(new_models)) * 100:.1f}%)"
                        )
                    return result
                except Exception as e:
                    logger.error(f"Failed to profile {model_info.name}: {e}")
                    return None
        
        tasks = [
            profile_single(model_info, i)
            for i, model_info in enumerate(new_models, 1)
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in task_results:
            if isinstance(r, ProfileResult):
                results.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Profiling task raised exception: {r}")
    else:
        for i, model_info in enumerate(new_models, 1):
            try:
                profiler = ModelProfiler(
                    client,
                    total_models=len(new_models),
                    current_model_num=i,
                    model_name=model_info.name,
                    model_size_bytes=model_info.size,
                )
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
