import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from router.backends.base import LLMBackend
from router.benchmark_db import (
    get_all_benchmarks,
    get_benchmarks_for_models,
    invalidate_all_caches,
    invalidate_profiles_cache,
)
from router.config import settings
from router.database import get_session
from router.models import ModelBenchmark, ModelFeedback, ModelProfile, RoutingDecision

logger = logging.getLogger(__name__)

_PROFILE_CACHE_TTL = 60.0
_profiles_cache: list[dict] | None = None
_profiles_cache_time: float = 0.0


@dataclass
class RoutingResult:
    selected_model: str
    confidence: float
    reasoning: str


# Minimum model sizes (in billions) required for task complexity
# If a model is below the minimum for its category/complexity, it gets a severe penalty
CATEGORY_MIN_SIZES = {
    "coding": {"simple": 0, "medium": 4, "hard": 8},
    "reasoning": {"simple": 0, "medium": 4, "hard": 8},
    "creativity": {"simple": 0, "medium": 1, "hard": 4},
    "general": {"simple": 0, "medium": 1, "hard": 4},
}


class SemanticCache:
    """Smart LRU cache for routing decisions with semantic similarity and response caching.
    
    Optimized with numpy for vectorized similarity search and separate locks for
    routing and response caches to minimize contention.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.85,
        embed_model: str | None = None,
        response_max_size: int = 50,
        embedding_ttl_seconds: int = 86400,  # 24 hours for embeddings
    ):
        self.cache: OrderedDict[str, tuple[RoutingResult, float, list[float] | None, float | None]] = (
            OrderedDict()
        )
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.embed_model = embed_model
        self.recent_selections: list[tuple[str, float]] = []
        self.max_recent = 20
        self._routing_lock = asyncio.Lock()
        self._response_lock = asyncio.Lock()
        self._embedding_lock = asyncio.Lock()

        self.response_cache: OrderedDict[tuple[str, str], tuple[str, float]] = OrderedDict()
        self.response_max_size = response_max_size
        self.response_ttl = ttl_seconds
        
        # Separate embedding cache with longer TTL (24h default)
        # Key: prompt hash, Value: (embedding, magnitude, timestamp)
        self.embedding_cache: dict[str, tuple[list[float], float, float]] = {}
        self.embedding_ttl = embedding_ttl_seconds

        self.stats = {
            "routing_hits": 0,
            "routing_misses": 0,
            "routing_similarity_hits": 0,
            "response_hits": 0,
            "response_misses": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
        }
        
        self._model_frequency: dict[str, int] = {}

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    def _cosine_similarity(self, a: list[float], b: list[float], mag_a: float, mag_b: float) -> float:
        if not a or not b or mag_a == 0 or mag_b == 0:
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        return dot_product / (mag_a * mag_b)

    def _cosine_similarity_batch(
        self, 
        query: list[float], 
        query_mag: float,
        candidates: list[tuple[str, list[float], float]]
    ) -> list[tuple[str, float]]:
        """Vectorized batch cosine similarity using numpy for better performance.
        
        Args:
            query: Query embedding vector
            query_mag: Magnitude of query vector
            candidates: List of (key, embedding, magnitude) tuples
            
        Returns:
            List of (key, similarity) tuples sorted by similarity descending
        """
        if not candidates:
            return []
        
        try:
            import numpy as np
            query_arr = np.array(query, dtype=np.float32)
            candidate_matrix = np.array([c[1] for c in candidates], dtype=np.float32)
            mags = np.array([c[2] for c in candidates], dtype=np.float32)
            keys = [c[0] for c in candidates]
            
            dot_products = candidate_matrix @ query_arr
            similarities = dot_products / (mags * query_mag + 1e-10)
            
            results = [(keys[i], float(similarities[i])) for i in range(len(keys))]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except ImportError:
            results = [
                (c[0], self._cosine_similarity(query, c[1], query_mag, c[2]))
                for c in candidates
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    async def _get_embedding(self, client: LLMBackend, text: str) -> list[float] | None:
        """Get embedding for text, using cache if available.
        
        Embeddings are cached for 24 hours by default to avoid repeated
        expensive embedding API calls for the same prompts.
        """
        if not self.embed_model:
            return None
        
        key = self._hash_prompt(text[:8192])
        current_time = time.time()
        
        async with self._embedding_lock:
            if key in self.embedding_cache:
                emb, mag, timestamp = self.embedding_cache[key]
                if current_time - timestamp < self.embedding_ttl:
                    self.stats["embedding_cache_hits"] += 1
                    return emb
                else:
                    del self.embedding_cache[key]
        
        try:
            result = await client.embed(self.embed_model, text[:8192])
            embeddings = result.get("embeddings") or result.get("embedding")
            if embeddings:
                emb = embeddings[0] if isinstance(embeddings[0], list) else embeddings
                if embeddings:
                    emb = embeddings[0] if isinstance(embeddings[0], list) else embeddings[0] if embeddings else None
                    if emb:
                        mag = sum(x * x for x in emb) ** 0.5
                        async with self._embedding_lock:
                            self.embedding_cache[key] = (emb, mag, current_time)
                        self.stats["embedding_cache_misses"] += 1
                        return emb
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
        return None

    async def get(self, prompt: str, embedding: list[float] | None = None) -> RoutingResult | None:
        key = self._hash_prompt(prompt)
        current_time = time.time()

        async with self._routing_lock:
            if key in self.cache:
                result, timestamp, _, _ = self.cache[key]
                if current_time - timestamp < self.ttl:
                    self.cache.move_to_end(key)
                    self.stats["routing_hits"] += 1
                    logger.debug(f"Cache hit (exact) for prompt hash: {key[:8]}...")
                    return result
                else:
                    del self.cache[key]

            if embedding:
                embedding_mag = sum(x * x for x in embedding) ** 0.5
                candidates = [
                    (cache_key, cache_emb, cache_mag)
                    for cache_key, (_, timestamp, cache_emb, cache_mag) in self.cache.items()
                    if cache_emb and cache_mag and current_time - timestamp < self.ttl
                ]
                
                if candidates:
                    similarities = self._cosine_similarity_batch(
                        embedding, embedding_mag, candidates
                    )
                    
                    for cache_key, similarity in similarities:
                        if similarity >= self.similarity_threshold:
                            result = self.cache[cache_key][0]
                            self.cache.move_to_end(cache_key)
                            self.stats["routing_similarity_hits"] += 1
                            logger.debug(
                                f"Cache hit (similarity={similarity:.2f}) for prompt hash: {cache_key[:8]}..."
                            )
                            return result

            self.stats["routing_misses"] += 1
            return None

    async def set(
        self,
        prompt: str,
        result: RoutingResult,
        embedding: list[float] | None = None,
    ) -> None:
        key = self._hash_prompt(prompt)
        async with self._routing_lock:
            mag = sum(x * x for x in embedding) ** 0.5 if embedding else None
            self.cache[key] = (result, time.time(), embedding, mag)
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            self.recent_selections.append((result.selected_model, time.time()))
            if len(self.recent_selections) > self.max_recent:
                old = self.recent_selections.pop(0)
                if old[0] in self._model_frequency:
                    self._model_frequency[old[0]] = max(0, self._model_frequency[old[0]] - 1)
            
            self._model_frequency[result.selected_model] = self._model_frequency.get(result.selected_model, 0) + 1
            
        logger.debug(f"Cached routing decision for: {key[:8]}...")

    def _make_response_key(self, model: str, prompt: str, params: dict | None = None) -> tuple:
        """Create cache key including model, prompt, and generation parameters."""
        prompt_hash = self._hash_prompt(prompt)
        if params:
            # Include relevant generation parameters that affect output
            param_tuple = tuple(sorted([
                (k, v) for k, v in params.items()
                if v is not None and k in ('temperature', 'top_p', 'max_tokens', 'seed', 'presence_penalty', 'frequency_penalty')
            ]))
            return (model, prompt_hash, param_tuple)
        return (model, prompt_hash)

    async def get_response(self, model: str, prompt: str, params: dict | None = None) -> str | None:
        key = self._make_response_key(model, prompt, params)
        current_time = time.time()

        async with self._response_lock:
            if key in self.response_cache:
                response, timestamp = self.response_cache[key]
                if current_time - timestamp < self.response_ttl:
                    self.response_cache.move_to_end(key)
                    self.stats["response_hits"] += 1
                    logger.debug(f"Response cache hit for {model}")
                    return response
                else:
                    del self.response_cache[key]

            self.stats["response_misses"] += 1
            return None

    async def set_response(self, model: str, prompt: str, response: str, params: dict | None = None) -> None:
        key = self._make_response_key(model, prompt, params)
        async with self._response_lock:
            self.response_cache[key] = (response, time.time())
            self.response_cache.move_to_end(key)
            if len(self.response_cache) > self.response_max_size:
                self.response_cache.popitem(last=False)
        logger.debug(f"Cached response for {model}")

    async def invalidate_response(self, model: str | None = None) -> int:
        count = 0
        async with self._response_lock:
            if model is None:
                count = len(self.response_cache)
                self.response_cache.clear()
            else:
                keys_to_remove = [k for k in self.response_cache if k[0] == model]
                count = len(keys_to_remove)
                for k in keys_to_remove:
                    del self.response_cache[k]
        return count

    async def get_model_frequency(self, model_name: str) -> float:
        async with self._routing_lock:
            if not self._model_frequency:
                return 0.0
            total = sum(self._model_frequency.values())
            if total == 0:
                return 0.0
            return self._model_frequency.get(model_name, 0) / total

    async def get_stats(self) -> dict[str, Any]:
        async with self._routing_lock:
            routing_stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["routing_hits"],
                "similarity_hits": self.stats["routing_similarity_hits"],
                "misses": self.stats["routing_misses"],
            }
            total_routing = self.stats["routing_hits"] + self.stats["routing_misses"]
            routing_stats["hit_rate"] = round(
                self.stats["routing_hits"] / total_routing if total_routing > 0 else 0.0, 3
            )
            
        async with self._response_lock:
            response_stats = {
                "size": len(self.response_cache),
                "max_size": self.response_max_size,
                "hits": self.stats["response_hits"],
                "misses": self.stats["response_misses"],
            }
            total_response = self.stats["response_hits"] + self.stats["response_misses"]
            response_stats["hit_rate"] = round(
                self.stats["response_hits"] / total_response if total_response > 0 else 0.0, 3
            )

        return {
            "routing": routing_stats,
            "response": response_stats,
        }


class RouterEngine:
    def __init__(
        self,
        client: LLMBackend,
        dispatcher_model: str | None = None,
        cache_enabled: bool = True,
        cache_max_size: int = 500,
        cache_ttl_seconds: int = 3600,
        cache_similarity_threshold: float = 0.85,
        cache_response_max_size: int = 200,
        embed_model: str | None = None,
        vram_manager: Any | None = None,  # VRAMManager, using Any to avoid circular import
    ):
        self.client = client
        self.dispatcher_model = dispatcher_model or settings.router_model
        self.cache_enabled = cache_enabled
        self.embed_model = embed_model
        self.vram_manager = vram_manager
        self.semantic_cache: SemanticCache | None

        if cache_enabled:
            self.semantic_cache = SemanticCache(
                max_size=cache_max_size,
                ttl_seconds=cache_ttl_seconds,
                similarity_threshold=cache_similarity_threshold,
                embed_model=embed_model,
                response_max_size=cache_response_max_size,
            )
        else:
            self.semantic_cache = None

    def warmup_caches(self, model_names: list[str] | None = None) -> None:
        """Pre-warm caches on startup to avoid first-request latency."""
        from router.benchmark_db import get_benchmarks_for_models

        self._get_all_profiles()
        if model_names:
            get_benchmarks_for_models(model_names)
        logger.info("Router caches pre-warmed")

    def invalidate_caches(self) -> None:
        """Invalidate all caches (call when models change)."""
        invalidate_all_caches()
        logger.info("Router caches invalidated")

    async def select_model(
        self, prompt: str | list[dict], request_obj: Any = None
    ) -> RoutingResult:
        prompt_str = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)

        # Always attempt cache lookup when enabled - exact hash works without embedding model
        embedding: list[float] | None = None
        if self.cache_enabled and self.semantic_cache:
            if self.embed_model:
                embedding = await self.semantic_cache._get_embedding(self.client, prompt_str)
            cached = await self.semantic_cache.get(prompt_str, embedding)
            if cached:
                return cached

        available_models = await self.client.list_models()
        if not available_models:
            raise ValueError("No models available")

        model_names = [m.name for m in available_models]

        # Convert prompt to string for analysis
        if isinstance(prompt, str):
            text_prompt = prompt
        else:
            text_parts: list[str] = []
            for msg in prompt:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
            text_prompt = "\n".join(text_parts)

        if self.dispatcher_model:
            result = await self._llm_dispatch(text_prompt, model_names)
        else:
            result = await self._keyword_dispatch(text_prompt, model_names, request_obj)

        if self.cache_enabled and self.semantic_cache:
            await self.semantic_cache.set(prompt_str, result, embedding)

        return result

    async def _llm_dispatch(self, prompt: str, model_names: list[str]) -> RoutingResult:
        # ... existing implementation ... (no changes needed here for now as it's string based)
        benchmarks = get_benchmarks_for_models(model_names)

        if not benchmarks:
            logger.warning("No benchmark data, falling back to keyword dispatch")
            return await self._keyword_dispatch(prompt, model_names)

        if not self.dispatcher_model:
            logger.warning("No dispatcher model configured, falling back to keyword dispatch")
            return await self._keyword_dispatch(prompt, model_names)

        context = self._build_dispatch_context(benchmarks)

        dispatch_prompt = f"""You are a model router. Given the user prompt and the available models with their benchmark scores, select the best model.

Available models:
{context}

User prompt: {prompt}

Respond ONLY with a JSON object in this exact format:
{{"model": "model_name", "reasoning": "brief explanation"}}

Select the model that best matches the user's prompt needs."""

        try:
            response = await self.client.chat(
                model=self.dispatcher_model,
                messages=[{"role": "user", "content": dispatch_prompt}],
                temperature=settings.router_temperature,
                max_tokens=settings.router_max_tokens,
            )

            content = response.get("message", {}).get("content", "")
            result = self._parse_llm_response(content, model_names)

            if result:
                return RoutingResult(
                    selected_model=result["model"],
                    confidence=0.9,
                    reasoning=result["reasoning"],
                )

        except Exception as e:
            logger.warning(f"LLM dispatch failed: {e}, falling back to keyword dispatch")

        return await self._keyword_dispatch(prompt, model_names)

    def _build_dispatch_context(self, benchmarks: list[dict]) -> str:
        """Build context string for LLM dispatcher with benchmark data."""
        context_lines = []
        for bm in benchmarks:
            name = bm.get("ollama_name", "unknown")
            elo = bm.get("elo_rating", "N/A")
            reasoning = bm.get("reasoning_score", "N/A")
            coding = bm.get("coding_score", "N/A")
            context_lines.append(f"- {name}: ELO={elo}, Reasoning={reasoning}, Coding={coding}")
        return "\n".join(context_lines) if context_lines else "No benchmark data available"

    def _parse_llm_response(self, content: str, model_names: list[str]) -> dict | None:
        """Parse LLM response to extract model selection."""
        import json

        try:
            data = json.loads(content)
            model = data.get("model", "")
            reasoning = data.get("reasoning", "")

            # Validate model exists
            if model in model_names:
                return {"model": model, "reasoning": reasoning}

            # Try fuzzy match
            for name in model_names:
                if model.lower() in name.lower() or name.lower() in model.lower():
                    return {"model": name, "reasoning": reasoning}

        except json.JSONDecodeError:
            pass

        # Try extracting from text
        for name in model_names:
            if name in content:
                return {"model": name, "reasoning": "Extracted from response"}

        return None

    # ... existing methods ...

    def _get_model_feedback_scores(self) -> dict[str, float]:
        """Get average feedback score for each model."""
        if not settings.feedback_enabled:
            return {}

        try:
            with get_session() as session:
                from sqlalchemy import func
                # Use SQL aggregation for O(1) database load instead of O(N) full table scan
                results = (
                    session.query(
                        ModelFeedback.model_name,
                        func.avg(ModelFeedback.score).label('avg_score')
                    )
                    .group_by(ModelFeedback.model_name)
                    .all()
                )
                return {name: float(avg_score) for name, avg_score in results}
        except Exception as e:
            logger.warning(f"Failed to fetch feedback scores: {e}")
            return {}

    async def _keyword_dispatch(
        self, prompt: str, model_names: list[str], request_obj: Any = None
    ) -> RoutingResult:
        profiles = self._get_all_profiles()
        benchmarks = get_benchmarks_for_models(model_names)
        feedback_scores = self._get_model_feedback_scores()

        if not profiles and not benchmarks:
            logger.warning("No profiles or benchmarks found, selecting first available model")
            return RoutingResult(
                selected_model=model_names[0],
                confidence=0.0,
                reasoning="No profiling data available, defaulting to first model",
            )

        analysis = self._analyze_prompt(prompt, request_obj)
        logger.info(f"Prompt analysis: {analysis}")

        # Gather model selection frequencies for diversity penalty if cache enabled
        model_frequencies: dict[str, float] = {}
        if self.semantic_cache:
            freq_tasks = [self.semantic_cache.get_model_frequency(m) for m in model_names]
            freq_results = await asyncio.gather(*freq_tasks)
            model_frequencies = {m: f for m, f in zip(model_names, freq_results)}

        scores = self._calculate_combined_scores(
            profiles, benchmarks, analysis, model_names, feedback_scores, model_frequencies
        )

        # Log all scores for debugging
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top5 = [
            (
                m,
                round(s["score"], 2),
                round(s.get("base_score", 0), 2),
                s.get("coding", 0),
                s.get("creativity", 0),
            )
            for m, s in sorted_scores[:8]
        ]
        logger.info(f"Model scores (top 8): {top5}")
        logger.info(f"  (format: model, total_score, base_score, coding, creativity)")

        # Determine dominant category (threshold > 0.5) - but exclude complexity!
        task_categories = {k: v for k, v in analysis.items() if k != "complexity"}
        top_category = max(task_categories.items(), key=lambda x: x[1])
        dominant_category = top_category[0] if top_category[1] > 0.5 else None

        # STRICT FILTERING for Vision and Tools
        candidates_filter = set(model_names)

        # 1. Vision Filter
        if analysis.get("vision", 0) > 0:
            # Filter for models that support vision (llava, pixtral, gpt-4o, etc)
            vision_models = {
                name
                for name in model_names
                if "llava" in name.lower()
                or "pixtral" in name.lower()
                or "vision" in name.lower()
                or "gpt-4o" in name.lower()
                or "claude-3" in name.lower()
                or "gemini" in name.lower()
                or "minicpm" in name.lower()
                or "moondream" in name.lower()
            }
            if vision_models:
                candidates_filter &= vision_models
                logger.info(f"Vision detected. Filtering candidates: {candidates_filter}")
            else:
                logger.warning("Vision task detected but no vision models found!")

        # 2. Tool Calling Filter
        if analysis.get("tools", 0) > 0:
            # Filter for models good at tool use
            tool_models = {
                name
                for name in model_names
                if any(
                    kw in name.lower()
                    for kw in [
                        "gpt-4",
                        "claude-3",
                        "mistral-large",
                        "qwen2.5",
                        "llama3.1",
                        "command-r",
                        "hermes",
                    ]
                )
            }
            # Fallback: if no specific tool models, allow all but warn
            if tool_models:
                candidates_filter &= tool_models
                logger.info(f"Tool use detected. Filtering candidates: {candidates_filter}")

        if not candidates_filter:
            # If we filtered everything out, reset to all models
            logger.warning("Strict filtering removed all models. Resetting to full list.")
            candidates_filter = set(model_names)

        # Filter scores dict to only include candidates
        scores = {k: v for k, v in scores.items() if k in candidates_filter}

        if dominant_category:
            # ... existing logic ...
            max_cat_score = max(s[dominant_category] for s in scores.values())
            threshold = max_cat_score * 0.85

            candidates = {m: s for m, s in scores.items() if s[dominant_category] >= threshold}

            best_model_name = max(candidates.items(), key=lambda x: x[1]["score"])[0]
            confidence = candidates[best_model_name]["score"]
            reasoning = f"Selected top {dominant_category} model (score: {candidates[best_model_name][dominant_category]:.2f}) with best overall traits"
        else:
            # Balanced/General task - use overall weighted score
            best_model_name = max(scores.items(), key=lambda x: x[1]["score"])[0]
            confidence = scores[best_model_name]["score"]
            reasoning = self._build_reasoning(analysis, scores[best_model_name])

        # ... rest of method ...
        return RoutingResult(
            selected_model=best_model_name,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _get_all_profiles(self) -> list[dict]:
        global _profiles_cache, _profiles_cache_time
        now = time.monotonic()
        if _profiles_cache is not None and (now - _profiles_cache_time) < _PROFILE_CACHE_TTL:
            return _profiles_cache

        with get_session() as session:
            profiles = session.execute(select(ModelProfile)).scalars().all()
            _profiles_cache = [
                {
                    "name": p.name,
                    "reasoning": p.reasoning,
                    "coding": p.coding,
                    "creativity": p.creativity,
                    "speed": p.speed,
                    "avg_response_time_ms": p.avg_response_time_ms,
                    "first_seen": p.first_seen,
                }
                for p in profiles
            ]
            _profiles_cache_time = now
            return _profiles_cache

    def _calculate_combined_scores(
        self,
        profiles: list[dict],
        benchmarks: list[dict],
        analysis: dict[str, float],
        model_names: list[str],
        feedback_scores: dict[str, float] = {},
        model_frequencies: dict[str, float] = {},
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = {}

        profile_map = {p["name"]: p for p in profiles}
        benchmark_map = {b["ollama_name"]: b for b in benchmarks}

        # Quality vs Speed Trade-off
        quality_pref = settings.quality_preference
        speed_weight = 1.0 - quality_pref
        quality_weight = quality_pref + 0.5  # Boost quality signals if preferred

        normalized_benchmark_map = {}
        for name in model_names:
            # Extract base name - handle versions and quantizations
            base = name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")

            # Also try with just the first part before numbers
            base_parts = base.split("2")[0] if "2" in base else base

            best_match = None
            best_score = 0.0

            for bm_name, bm in benchmark_map.items():
                bm_base = (
                    bm_name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
                )

                # Exact match
                if base == bm_base:
                    best_match = bm
                    best_score = 100.0
                    break

                # Partial match - check if major model name matches
                # e.g., "qwen2.5" matches "qwen2.5coder" or "qwen"
                if base in bm_base or bm_base in base:
                    score = len(base) / max(len(base), len(bm_base), 1)
                    if score > best_score:
                        best_match = bm
                        best_score = score
                elif any(part in bm_base for part in base.split() if len(part) > 2):
                    # Try matching individual parts
                    for part in [base, base[:4], base[:6]]:
                        if part in bm_base and len(part) > 2:
                            best_match = bm
                            best_score = 0.5
                            break

            if best_match:
                normalized_benchmark_map[name] = best_match

        logger.info(
            f"Benchmark matching: {len(normalized_benchmark_map)}/{len(model_names)} models matched"
        )

        # Log benchmark match details for each model
        for name in model_names:
            bench_match = normalized_benchmark_map.get(name)
            if bench_match:
                logger.info(
                    f"  {name} -> benchmark: reasoning={bench_match.get('reasoning_score')}, coding={bench_match.get('coding_score')}, elo={bench_match.get('elo_rating')}"
                )
            else:
                logger.info(f"  {name} -> NO benchmark match")

        # Build model category affinity based on model name patterns
        model_category_affinity = self._build_model_category_affinity(
            model_names, normalized_benchmark_map
        )

        # Determine dominant category (exclude complexity which is a meta-category)
        task_categories = {k: v for k, v in analysis.items() if k not in ("complexity", "vision", "tools")}
        if task_categories:
            top_category = max(task_categories.items(), key=lambda x: x[1])
            dominant_category = top_category[0] if top_category[1] > 0.5 else None
        else:
            dominant_category = None

        for model_name in model_names:
            profile = profile_map.get(model_name)
            benchmark = normalized_benchmark_map.get(model_name)
            affinity = model_category_affinity.get(model_name, {})

            base_score = 0.0

            # Map prompt categories to benchmark/profile fields
            category_map = {
                "reasoning": ("reasoning_score", "reasoning"),
                "coding": ("coding_score", "coding"),
                "creativity": ("creativity", None),  # No benchmark creativity, use profile only
                "factual": ("general_score", "factual"),  # Use general_score for factual
            }

            for category, weight in analysis.items():
                # Skip complexity - it's handled separately as a bonus
                if category == "complexity":
                    continue

                # Signal 1: Precise Benchmarks (MMLU, HumanEval, etc.)
                benchmark_score = 0.0
                if benchmark:
                    bm_field, _ = category_map.get(category, (None, None))
                    benchmark_score = benchmark.get(bm_field, 0.0) or 0.0
                    # Convert 0-100 scale to 0.0-1.0
                    if benchmark_score > 1.0:
                        benchmark_score /= 100.0

                # Signal 2: General Quality (ELO / Arena Score)
                elo_signal = 0.0
                if benchmark and benchmark.get("elo_rating"):
                    raw_elo = benchmark["elo_rating"]
                    if raw_elo > 200:  # True ELO
                        elo_signal = max(min((raw_elo - 1000) / 800, 1.5), 0.0)
                    else:  # 0-100 Score
                        elo_signal = raw_elo / 100.0

                # Signal 3: Name-based Inference (Fallback)
                inference_score = affinity.get(category, 0.0)

                # Signal 4: Profile Scores (Runtime profiling data)
                profile_score = 0.0
                if profile:
                    _, profile_field = category_map.get(category, (None, None))
                    if profile_field:
                        profile_score = profile.get(profile_field, 0.0) or 0.0

                # Weighted combination of signals
                combined_cat_score = (
                    (benchmark_score * 1.5 * quality_weight)
                    + (elo_signal * 1.0 * quality_weight)
                    + (inference_score * 0.4 * quality_weight)
                    + (profile_score * 0.8 * quality_weight)  # Profile is more reliable than name inference
                )

                # If this is the dominant category, apply the 20x Category-First boost
                # BUT only if we have actual data (benchmark, ELO, or profile)
                has_actual_data = benchmark_score > 0 or elo_signal > 0 or profile_score > 0

                # Check if model meets minimum size for this category + complexity
                complexity_bucket = self._get_complexity_bucket(analysis.get("complexity", 0.0))
                min_size = CATEGORY_MIN_SIZES.get(category, {}).get(complexity_bucket, 0)
                params = self._extract_parameter_count(model_name)
                has_adequate_size = params is not None and params >= min_size

                if category == dominant_category and combined_cat_score > 0.15:
                    if has_actual_data:
                        combined_cat_score *= 20.0  # Strong boost with data
                    elif has_adequate_size:
                        combined_cat_score *= (
                            10.0  # Moderate boost with adequate size but no benchmark
                        )
                    else:
                        combined_cat_score *= 1.5  # Weak boost without data or size

                if weight > 0:
                    base_score += combined_cat_score * weight
                else:
                    base_score += combined_cat_score * 0.01

            # Bonus factors (speed, size, newness, complexity, feedback)
            bonus_score = 0.0
            params = self._extract_parameter_count(model_name)
            complexity = analysis.get("complexity", 0.0)
            has_benchmark = normalized_benchmark_map.get(model_name) is not None

            # Feedback Bonus
            fb_score = feedback_scores.get(model_name, 0.0)
            if fb_score != 0:
                bonus_score += fb_score * 2.0  # Significant impact for user preference

            # Bonus for having benchmark data (prefer data-driven over name-based)
            if has_benchmark:
                bonus_score += 0.3 * quality_weight

            # === SIZE/CAPACITY SCORING ===
            # Only apply size bonus when we have benchmark data OR high complexity
            # This prevents size from dominating when we only have runtime profiles (speed-based routing)
            size_score = 0.0
            if params:
                if params >= 30:
                    size_score = 3.0 * quality_weight  # Strong preference for very large models
                elif params >= 14:
                    size_score = 2.0 * quality_weight  # Good preference for large models
                elif params >= 7:
                    size_score = 1.0 * quality_weight  # Medium preference for mid-size models
                elif params >= 3:
                    size_score = 0.0  # Neutral for small models
                else:
                    # Penalize tiny models less if we prefer speed (low quality_pref)
                    penalty = -2.0 if quality_pref >= 0.5 else -0.5
                    size_score = penalty

            # Apply size score only for moderate+ complexity tasks
            if complexity >= 0.3:
                bonus_score += size_score * 0.5

            # Complexity-Size Matching Logic (enhanced)
            if complexity >= 0.5:
                # Very high complexity: STRONGLY prefer larger models
                if params and params >= 30:
                    bonus_score += 3.0 * quality_weight
                elif params and params >= 14:
                    bonus_score += 2.0 * quality_weight
                elif params and params >= 7:
                    bonus_score += 1.0 * quality_weight
                elif params and params < 4:
                    bonus_score -= 3.0 * quality_weight
            elif complexity >= 0.3:
                # Moderate complexity: Prefer larger models
                if params and params >= 30:
                    bonus_score += 2.0 * quality_weight
                elif params and params >= 14:
                    bonus_score += 1.2 * quality_weight
                elif params and params >= 7:
                    bonus_score += 0.4 * quality_weight
                elif params and params < 4:
                    bonus_score -= 2.0 * quality_weight
            elif complexity < 0.15:
                # Low complexity: Strong preference for small/fast models
                if params and params <= 4:
                    bonus_score += 1.5 * speed_weight  # Strong bonus for tiny models
                elif params and params <= 7:
                    bonus_score += 0.8 * speed_weight  # Good bonus for small models
                elif params and params >= 14:
                    bonus_score -= 1.0 * speed_weight  # Penalize large models
                elif params and params >= 30:
                    bonus_score -= 2.0 * speed_weight  # Strong penalty for very large

            # === CATEGORY-AWARE MINIMUM SIZE REQUIREMENTS ===
            # Apply severe penalty if model is below minimum size for category + complexity
            if dominant_category and params is not None:
                complexity_bucket = self._get_complexity_bucket(complexity)
                min_size = CATEGORY_MIN_SIZES.get(dominant_category, {}).get(complexity_bucket, 0)

                if params < min_size:
                    # Calculate deficit - scales with how far below minimum
                    size_deficit = min_size - params
                    min_size_penalty = -10.0 * size_deficit
                    bonus_score += min_size_penalty
                    logger.debug(
                        f"Min size penalty for {model_name}: {min_size_penalty} (params={params}, min={min_size} for {dominant_category}/{complexity_bucket})"
                    )

            if profile:
                # Speed bonus (only for simple tasks OR if speed is preferred)
                # Boost speed importance if quality_pref is low
                speed_importance = speed_weight * 2.0

                if (complexity < 0.4 or speed_importance > 1.0) and profile.get(
                    "avg_response_time_ms", 0
                ) > 0:
                    # More sensitive time factor (baseline 10s instead of 60s)
                    time_factor = 1.0 - min(profile["avg_response_time_ms"] / 10000.0, 0.8)
                    bonus_score += time_factor * 0.2 * speed_importance

                # Newness bonus
                if settings.prefer_newer_models and profile.get("first_seen"):
                    newness = self._calculate_newness_score(profile["first_seen"])
                    bonus_score += newness * 0.05

            # Diversity Penalty: Reduce score if model has been selected too frequently recently
            # This prevents one model from dominating and encourages exploration
            model_frequency = model_frequencies.get(model_name, 0.0)
            diversity_penalty = 0.0
            
            if model_frequency > 0.5:
                # Apply multiplicative penalty to base_score
                # freq 0.5 -> 0.65x (35% reduction)
                # freq 0.8 -> 0.44x (56% reduction)
                # freq 1.0 -> 0.30x (70% reduction)
                reduction = (model_frequency - 0.5) * 1.4  # scales from 0 to 0.7
                multiplier = max(0.3, 1.0 - reduction)
                base_score = base_score * multiplier
                # For logging, compute an approximate additive equivalent
                diversity_penalty = -base_score * (1.0 - multiplier) / (multiplier if multiplier > 0 else 1)
            elif model_frequency > 0:
                # Small frequency gets tiny penalty to nudge exploration
                diversity_penalty = -model_frequency * 0.2
                # No multiplicative penalty for low frequency

            total_score = base_score + bonus_score + diversity_penalty

            # Use the actual scores used for routing in the debug log
            scores[model_name] = {
                "score": total_score,
                "base_score": base_score,
                "bonus": bonus_score,
                "diversity": diversity_penalty,
                "reasoning": affinity.get("reasoning", 0),
                "coding": affinity.get("coding", 0),
                "creativity": affinity.get("creativity", 0),
                "factual": affinity.get("factual", 0),
            }

        # Debug: log all model scores
        logger.info(
            f"Actual routing affinity scores: {[(m, s.get('reasoning', 0), s.get('coding', 0), s.get('creativity', 0)) for m, s in scores.items()]}"
        )

        return scores

    def _extract_parameter_count(self, model_name: str) -> float | None:
        """Extract parameter count in billions from model name."""
        name_lower = model_name.lower()

        # 1. Direct Regex (e.g., "7b", "0.5b", "1.5b") - check BEFORE colon
        # First, get the part before the colon (the model tag)
        model_tag = name_lower.split(":")[0] if ":" in name_lower else name_lower

        match = re.search(r"(\d+(\.\d+)?)\s*b", model_tag)
        if match:
            return float(match.group(1))

        # 2. Known model size mappings for Ollama names
        size_map = {
            "mini": 3.8,  # Phi-3-mini
            "small": 7.0,  # Mistral-small, etc
            "medium": 14.0,  # Phi-3-medium, etc
            "large": 70.0,
            "nemo": 12.0,  # Mistral-Nemo
            "r1": 14.0,  # DeepSeek-R1 (common Ollama default is 14B)
            "gemma3": 1.0,  # Gemma 3 is 1B
            "gemma2": 9.0,  # Gemma 2 is 9B
        }

        for key, size in size_map.items():
            if key in name_lower:
                return size

        # 3. Handle names like "llama3.1" (default is 8b)
        if "llama3" in name_lower or "llama3.1" in name_lower or "llama3.2" in name_lower:
            if ":1b" in name_lower or "1b" in model_tag:
                return 1.0
            if ":3b" in name_lower or "3b" in model_tag:
                return 3.0
            if ":8b" in name_lower or "8b" in model_tag:
                return 8.0
            return 8.0  # default

        if "qwen2.5" in name_lower:
            if ":0.5b" in name_lower or "0.5b" in model_tag:
                return 0.5
            if ":1.5b" in name_lower or "1.5b" in model_tag:
                return 1.5
            if ":7b" in name_lower or "7b" in model_tag:
                return 7.0
            if ":14b" in name_lower or "14b" in model_tag:
                return 14.0
            if ":32b" in name_lower or "32b" in model_tag:
                return 32.0
            if ":72b" in name_lower or "72b" in model_tag:
                return 72.0

        return None

    def _get_complexity_bucket(self, complexity: float) -> str:
        """Determine complexity bucket based on complexity score."""
        if complexity < 0.2:
            return "simple"
        elif complexity < 0.5:
            return "medium"
        else:
            return "hard"

    def _calculate_size_score(self, params: float | None) -> float:
        """Calculate score based on model size (smaller is better)."""
        if params is None:
            return 0.5  # Neutral score if unknown

        # Logarithmic-ish scaling:
        # < 3B -> 1.0
        # 7-8B -> 0.8
        # 13-14B -> 0.6
        # 30B+ -> 0.4
        if params <= 3:
            return 1.0
        elif params <= 8:
            return 0.8
        elif params <= 14:
            return 0.6
        elif params <= 35:
            return 0.4
        else:
            return 0.2

    def _calculate_newness_score(self, first_seen) -> float:
        """Calculate score based on how new the model is to the system."""
        if not first_seen:
            return 0.0

        # Handle both timezone-aware and naive datetimes
        if isinstance(first_seen, datetime):
            if first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age = now - first_seen
        days_old = age.days

        # New models (< 1 day) get boost
        if days_old < 1:
            return 1.0
        elif days_old < 7:
            return 0.8
        elif days_old < 30:
            return 0.5
        else:
            return 0.0

    def _analyze_prompt(self, prompt: str, request_obj: Any = None) -> dict[str, float]:
        prompt_lower = prompt.lower()

        analysis: dict[str, float] = {
            "reasoning": 0.0,
            "coding": 0.0,
            "creativity": 0.0,
            "factual": 0.0,
            "complexity": 0.0,
            "vision": 0.0,  # New
            "tools": 0.0,  # New
        }

        # New: Inspect request object for capabilities
        if request_obj:
            # 1. Vision Detection
            if hasattr(request_obj, "messages"):
                for msg in request_obj.messages:
                    if isinstance(msg.content, list):
                        for part in msg.content:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                analysis["vision"] = 1.0
                                break

            # 2. Tool Detection
            if hasattr(request_obj, "tools") and request_obj.tools:
                analysis["tools"] = 1.0
                analysis["complexity"] += 0.3  # Tools imply complexity
                analysis["coding"] += 0.2  # Tools often relate to coding/structured output

            # 3. JSON Mode Detection
            if hasattr(request_obj, "response_format") and request_obj.response_format:
                if request_obj.response_format.get("type") == "json_object":
                    analysis["coding"] += 0.3  # JSON mode is coding-adjacent
                    analysis["complexity"] += 0.1

        # ... existing logic ...
        reasoning_keywords = [
            "calculate",
            "logic",
            "solve",
            "reason",
            "prove",
            "math",
            "sequence",
            "pattern",
            "if then",
            "therefore",
            "because",
            "derive",
            "speed",
            "velocity",
            "distance",
            "how much",
            "how many",
            "result",
        ]
        coding_keywords = [
            "code",
            "function",
            "implement",
            "algorithm",
            "program",
            "python",
            "javascript",
            "java",
            "sql",
            "debug",
            "api",
            "class",
            "def ",
            "return",
            "import",
            "write code",
            "bug",
            "fix",
            "script",
            "json",
            "xml",
            "yaml",
            "parse",
            "schema",  # Added data formats
        ]
        # ... rest of keyword analysis ...
        creative_keywords = [
            "story",
            "write",
            "poem",
            "creative",
            "imagine",
            "describe",
            "invent",
            "fantasy",
            "narrative",
            "character",
            "scene",
            "song",
            "haiku",
            "lyrics",
            "joke",
            "humor",
        ]
        factual_keywords = [
            "what is",
            "who is",
            "when did",
            "where is",
            "define",
            "explain",
            "fact",
            "history",
            "capital",
            "year",
            "date",
            "list",
            "tell me about",
            "summary",
            "summarize",
        ]

        for kw in reasoning_keywords:
            if kw in prompt_lower:
                analysis["reasoning"] += 0.3

        for kw in coding_keywords:
            if kw in prompt_lower:
                analysis["coding"] += 0.4

        for kw in creative_keywords:
            if kw in prompt_lower:
                analysis["creativity"] += 0.35

        for kw in factual_keywords:
            if kw in prompt_lower:
                analysis["factual"] += 0.3

        # Complexity Detection (Enhanced for Difficulty Prediction)
        # Length heuristics
        if len(prompt) > 500:
            analysis["complexity"] += 0.2
        if len(prompt) > 1500:
            analysis["complexity"] += 0.3

        # Structure heuristics
        if prompt.count("?") > 2:
            analysis["complexity"] += 0.1
        if prompt.count("\n") > 5:
            analysis["complexity"] += 0.1

        # Keyword-based complexity
        complexity_keywords = [
            "complex",
            "expert",
            "detailed",
            "comprehensive",
            "optimized",
            "architecture",
            "distributed",
            "performance",
            "scalable",
            "deep dive",
            "advanced",
            "professional",
            "senior",
            "production-ready",
            "implement",
            "algorithm",
            "data structure",
            "tree",
            "graph",
            "recursive",
            "unit test",
            "type hint",
            "generics",
            "async",
            "concurrent",
            "nuance",
            "subtle",
            "imply",
            "hidden meaning",
            "step-by-step",
            "reasoning chain",
        ]

        for kw in complexity_keywords:
            if kw in prompt_lower:
                analysis["complexity"] += 0.15  # Incremental boost

        # Additional complexity for coding tasks with multiple requirements
        if analysis["coding"] > 0.5:
            # Count coding-related keywords to gauge complexity
            coding_complexity_indicators = [
                "with",
                "include",
                "and",
                "also",
                "plus",
                "additionally",
                "operations",
                "methods",
                "functions",
                "classes",
                "interface",
                "inheritance",
                "generic",
                "template",
                "exception",
                "handle",
                "error",
                "security",
                "thread",
            ]
            indicator_count = sum(1 for ind in coding_complexity_indicators if ind in prompt_lower)
            if indicator_count >= 3:
                analysis["complexity"] += 0.3
            elif indicator_count >= 2:
                analysis["complexity"] += 0.15

        # Cap complexity at 1.0
        analysis["complexity"] = min(analysis["complexity"], 1.0)

        code_indicators = ["```", "def ", "function ", "const ", "let ", "var ", "class "]
        for ind in code_indicators:
            if ind in prompt:
                analysis["coding"] = 1.0
                break

        if max(analysis.values()) == 0.0:
            analysis["factual"] = 0.5

        return analysis

    def _build_reasoning(
        self,
        analysis: dict[str, float],
        scores: dict[str, float],
    ) -> str:
        top_category = max(analysis.items(), key=lambda x: x[1])
        category_name = top_category[0] if top_category[1] > 0 else "balanced"

        return f"Matched to {category_name} profile (score: {scores['score']:.2f})"

    def _build_model_category_affinity(
        self,
        model_names: list[str],
        benchmark_map: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        """Infers category affinity from model names when benchmarks are missing."""
        affinity: dict[str, dict[str, float]] = {}

        for name in model_names:
            name_lower = name.lower()
            # Start with a "Generalist Floor" - every model has some base capability
            scores = {"coding": 0.1, "reasoning": 0.1, "creativity": 0.1, "factual": 0.1}

            # Specialist Boosts: Only for models that explicitly mention these in their name
            if any(
                kw in name_lower
                for kw in ["coder", "starcoder", "codegeex", "codellama", "deepseek-coder"]
            ):
                scores["coding"] = 0.9
                scores["reasoning"] = 0.5  # Coders are usually good at logic too

            if any(kw in name_lower for kw in ["r1", "math", "logic", "thought", "reasoner"]):
                scores["reasoning"] = 1.0

            if any(kw in name_lower for kw in ["dolphin", "uncensored", "creative", "writer"]):
                scores["creativity"] = 0.8

            # Generalists (Llama, Mistral, Gemma, Phi) are good at everything,
            # especially factual and creative tasks
            if any(kw in name_lower for kw in ["llama", "mistral", "gemma", "phi", "qwen"]):
                scores["factual"] = 0.7
                scores["creativity"] = 0.6 if scores["creativity"] < 0.6 else scores["creativity"]

            affinity[name] = scores

        return affinity

    async def log_decision(
        self,
        prompt: str,
        selected: str,
        confidence: float,
        reasoning: str,
        response_id: str | None = None,
    ) -> None:
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

            with get_session() as session:
                decision = RoutingDecision(
                    prompt_hash=prompt_hash,
                    selected_model=selected,
                    confidence=confidence,
                    reasoning=reasoning,
                    response_id=response_id,
                )
                session.add(decision)
                session.commit()

            # Also cache the routing decision for future similar prompts
            result = RoutingResult(
                selected_model=selected,
                confidence=confidence,
                reasoning=reasoning,
            )
            if self.semantic_cache:
                await self.semantic_cache.set(prompt, result)

        except Exception as e:
            logger.debug(f"Failed to log routing decision: {e}")
