"""ArtificialAnalysis.ai benchmark data provider.

This provider fetches benchmark data from https://artificialanalysis.ai/
and integrates it into SmarterRouter's benchmark database.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import yaml

from router.providers.base import BenchmarkProvider

logger = logging.getLogger(__name__)

# ArtificialAnalysis API endpoints
AA_BASE_URL = "https://artificialanalysis.ai/api/v2"
AA_MODELS_ENDPOINT = "/data/llms/models"

# Rate limiting: 1000 requests per day on free tier
RATE_LIMIT_PER_DAY = 1000


class ArtificialAnalysisProvider(BenchmarkProvider):
    """Provider for ArtificialAnalysis.ai benchmark data."""

    def __init__(self):
        self.api_key = os.getenv("ROUTER_ARTIFICIAL_ANALYSIS_API_KEY")
        self.cache_ttl = int(
            os.getenv("ROUTER_ARTIFICIAL_ANALYSIS_CACHE_TTL", "86400")
        )
        self.mapping_file = os.getenv(
            "ROUTER_ARTIFICIAL_ANALYSIS_MODEL_MAPPING_FILE"
        )

        # Load model mapping from YAML if provided
        self.model_mapping: dict[str, str] = {}
        if self.mapping_file and os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, "r") as f:
                    mapping_data = yaml.safe_load(f)
                    self.model_mapping = mapping_data.get("mappings", {})
                logger.info(
                    f"Loaded {len(self.model_mapping)} model mappings from {self.mapping_file}"
                )
            except Exception as e:
                logger.warning(f"Failed to load model mapping file: {e}")

        # In-memory cache
        self._cache: dict[str, Any] = {}
        self._cache_timestamp: datetime | None = None
        self._requests_today = 0
        self._day_reset = datetime.now(timezone.utc).date()

    @property
    def name(self) -> str:
        return "artificial_analysis"

    async def fetch_data(self, ollama_models: list[str]) -> list[dict[str, Any]]:
        """Fetch benchmark data from ArtificialAnalysis and map to Ollama models."""
        if not self.api_key:
            logger.warning(
                "ArtificialAnalysis API key not set. Set ROUTER_ARTIFICIAL_ANALYSIS_API_KEY."
            )
            return []

        # Check cache first
        if self._is_cache_valid():
            logger.debug("Using cached ArtificialAnalysis data")
            return list(self._cache.values())

        # Reset daily counter if needed
        self._reset_daily_counter_if_needed()

        if self._requests_today >= RATE_LIMIT_PER_DAY:
            logger.warning(
                "ArtificialAnalysis daily rate limit reached, using cached data if available"
            )
            return list(self._cache.values()) if self._cache else []

        # Fetch from API
        headers = {"x-api-key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{AA_BASE_URL}{AA_MODELS_ENDPOINT}", headers=headers
                )
                response.raise_for_status()
                data = response.json()

                if data.get("status") != 200:
                    logger.error(
                        f"ArtificialAnalysis API returned status: {data.get('status')}"
                    )
                    return []

                models = data.get("data", [])
                logger.info(f"Fetched {len(models)} models from ArtificialAnalysis")
                
                # Debug: log all distinct model creators to see what's available
                if models and not hasattr(self, '_debug_logged'):
                    creators = {}
                    for m in models:
                        creator = m.get('model_creator', {}).get('slug', 'unknown')
                        name = m.get('name', '')
                        if creator not in creators:
                            creators[creator] = []
                        if len(creators[creator]) < 3:  # Store up to 3 examples per creator
                            creators[creator].append(name)
                    
                    # Log all creators with example model names
                    logger.info(f"AA all model creators: {creators}")
                    
                    # Also show any models that might match expected local names (case-insensitive)
                    local_names = ['llama', 'gemma', 'phi', 'mistral', 'qwen', 'mistralnemo']
                    matches = []
                    for m in models:
                        name = m.get('name', '').lower()
                        if any(local in name for local in local_names):
                            matches.append((m.get('id'), m.get('name'), m.get('model_creator', {}).get('slug')))
                    if matches:
                        logger.info(f"AA potential local model matches: {matches}")
                    
                    self._debug_logged = True

                # Process and map models
                results: list[dict[str, Any]] = []
                for model_data in models:
                    ollama_name = self._map_to_ollama_name(model_data, ollama_models)
                    if not ollama_name:
                        continue

                    benchmark = self._convert_to_benchmark_dict(
                        model_data, ollama_name
                    )
                    if benchmark:
                        results.append(benchmark)
                        # Also cache by AA ID for future lookups
                        self._cache[model_data["id"]] = benchmark

                self._cache_timestamp = datetime.now(timezone.utc)
                self._requests_today += 1

                logger.info(
                    f"Converted {len(results)} ArtificialAnalysis models to benchmark data"
                )
                return results

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Invalid ArtificialAnalysis API key")
            elif e.response.status_code == 429:
                logger.warning("ArtificialAnalysis rate limit exceeded")
            else:
                logger.error(f"ArtificialAnalysis API error: {e}")
            return list(self._cache.values()) if self._cache else []
        except Exception as e:
            logger.error(f"Error fetching ArtificialAnalysis data: {e}")
            return list(self._cache.values()) if self._cache else []

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still fresh."""
        if not self._cache_timestamp:
            return False
        return (datetime.utcnow() - self._cache_timestamp) < timedelta(
            seconds=self.cache_ttl
        )

    def _reset_daily_counter_if_needed(self):
        """Reset daily request counter on new day."""
        today = datetime.now(timezone.utc).date()
        if today > self._day_reset:
            self._requests_today = 0
            self._day_reset = today

    def _map_to_ollama_name(
        self, model_data: dict[str, Any], ollama_models: list[str]
    ) -> str | None:
        """Map ArtificialAnalysis model to SmarterRouter Ollama model name."""
        aa_id = model_data.get("id")  # May be missing in test fixtures
        aa_name = model_data.get("name", "")
        creator = model_data.get("model_creator", {}).get("name", "")

        # Strategy 1: Explicit mapping from YAML file (by ID or name)
        if aa_id and aa_id in self.model_mapping:
            mapped = self.model_mapping[aa_id]
            if mapped in ollama_models:
                return mapped
        if aa_name in self.model_mapping:
            mapped = self.model_mapping[aa_name]
            if mapped in ollama_models:
                return mapped

        # Strategy 2: Construct name as "creator/name" (lowercase)
        if creator:
            # Normalize creator: lowercase, remove spaces, special chars
            creator_slug = creator.lower().replace(" ", "-").replace("_", "-")
            # Normalize model name: lowercase, replace spaces/dots, keep numbers
            model_slug = (
                aa_name.lower()
                .replace(" ", "-")
                .replace(".", "")
                .replace("_", "-")
            )
            candidate = f"{creator_slug}/{model_slug}"
            if candidate in ollama_models:
                return candidate

        # Strategy 3: Try just the model name (some backends don't use prefixes)
        if aa_name in ollama_models:
            return aa_name

        # Could not map - log for user to add mapping
        logger.debug(
            f"Could not map AA model '{aa_name}' (creator: {creator}) to any known Ollama model"
        )
        return None

    def _convert_to_benchmark_dict(
        self, model_data: dict[str, Any], ollama_name: str
    ) -> dict[str, Any] | None:
        """Convert ArtificialAnalysis data to benchmark dictionary."""
        try:
            evals = model_data.get("evaluations", {})
            pricing = model_data.get("pricing", {})
            speed_tokens_per_sec = model_data.get("median_output_tokens_per_second")
            ttft = model_data.get("median_time_to_first_token_seconds")

            # Extract standard benchmark scores (0-1 scale)
            mmlu = evals.get("mmlu_pro")  # Use MMLU-Pro as mmlu score
            gpqa = evals.get("gpqa")
            livecodebench = evals.get("livecodebench")  # Coding benchmark
            math = evals.get("math_500")  # Use Math-500 as math score

            # Convert to 0-100 scale for internal use if needed?
            # Our ModelBenchmark expects 0-1 floats for scores.
            # We'll store as 0-1.

            # Compute capability scores using same formula as HuggingFace provider
            # Reasoning: mmlu, mmlu_pro, gpqa, math
            # Coding: humaneval (use livecodebench), plus maybe other coding benchmarks
            reasoning = 0.0
            coding = 0.0
            general = 0.0
            count = 0

            if mmlu is not None:
                reasoning += mmlu
                general += mmlu
                count += 2

            if gpqa is not None:
                reasoning += gpqa
                count += 1

            if math is not None:
                reasoning += math
                general += math * 0.5
                count += 2

            if livecodebench is not None:
                coding += livecodebench
                general += livecodebench
                count += 2

            # Avoid division by zero
            if count > 0:
                reasoning_score = round(reasoning / count, 3)
                # Coding score: divide by 2 if any coding benchmark exists (like HuggingFace)
                coding_score = round(coding / max(1, (1 if livecodebench is not None else 0) + 1), 3)
                general_score = round(general / count, 3)
            else:
                reasoning_score = 0.0
                coding_score = 0.0
                general_score = 0.0

            # Build benchmark dict
            benchmark = {
                "ollama_name": ollama_name,
                "mmlu": mmlu,
                "humaneval": livecodebench,  # Use livecodebench as humaneval proxy
                "math": math,
                "gpqa": gpqa,
                "reasoning_score": reasoning_score,
                "coding_score": coding_score,
                "general_score": general_score,
                "full_name": model_data.get("name"),
                "throughput": speed_tokens_per_sec,  # tokens/sec
                # extra_data stores all the AA-specific info that doesn't fit standard columns
                "extra_data": {
                    "artificial_analysis_id": model_data.get("id"),
                    "model_creator": model_data.get("model_creator", {}).get("name"),
                    "artificial_analysis_intelligence_index": evals.get(
                        "artificial_analysis_intelligence_index"
                    ),
                    "artificial_analysis_coding_index": evals.get(
                        "artificial_analysis_coding_index"
                    ),
                    "artificial_analysis_math_index": evals.get(
                        "artificial_analysis_math_index"
                    ),
                    "mmlu_pro": mmlu,
                    "livecodebench": livecodebench,
                    "math_500": math,
                    "gpqa": gpqa,
                    "time_to_first_token_seconds": ttft,
                },
            }

            # Remove None values from extra_data (keep key if present but None? maybe keep for transparency)
            # We'll keep them for completeness.

            return benchmark

        except Exception as e:
            logger.warning(
                f"Failed to convert ArtificialAnalysis data for {model_data.get('name')}: {e}"
            )
            return None
