import json
import logging
import re
from typing import Any

import httpx

from router.providers.base import BenchmarkProvider

logger = logging.getLogger(__name__)

# HuggingFace Datasets Server REST API endpoint for Open LLM Leaderboard
HF_DATASET_SERVER_URL = "https://datasets-server.huggingface.co/first-rows?dataset=open-llm-leaderboard%2Fresults&config=default&split=train"

# Reuse the mapping from before, or import it if we separate constants
OLLAMA_MODEL_MAPPING = {
    "llama3.1": ["Llama-3.1-8B", "Meta-Llama-3.1-8B", "Llama-3.1-70B"],
    "llama3": ["Llama-3-8B", "Meta-Llama-3-8B", "Llama-3-70B"],
    "llama2": ["Llama-2-7B", "Llama-2-13B", "Llama-2-70B"],
    "llama2-chat": ["Llama-2-7B-Chat", "Llama-2-13B-Chat"],
    "mistral": ["Mistral-7B-v0.1", "Mistral-7B-v0.2"],
    "mistral-large": ["Mistral-Large-Instruct-2407"],
    "mixtral": ["Mixtral-8x7B-v0.1", "Mixtral-8x22B"],
    "codellama": ["CodeLlama-7B", "CodeLlama-13B", "CodeLlama-34B"],
    "phi3": ["Phi-3-mini-4k", "Phi-3-small-128k", "Phi-3-medium-4k"],
    "phi4": ["Phi-4-mini", "Phi-4"],
    "qwen2": ["Qwen2-0.5B", "Qwen2-1.5B", "Qwen2-7B", "Qwen2-72B"],
    "qwen2.5": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B", "Qwen2.5-72B"],
    "qwen": ["Qwen-1.8B", "Qwen-7B", "Qwen-14B", "Qwen-72B"],
    "gemma2": ["gemma-2-2b", "gemma-2-9b", "gemma-2-27b"],
    "gemma": ["gemma-2b", "gemma-7b"],
    "deepseek-coder": ["deepseek-coder-6.7b", "deepseek-coder-33b"],
    "deepseek-llm": ["deepseek-llm-7b", "deepseek-llm-67b"],
    "deepseek-chat": ["DeepSeek-Chat-7B", "DeepSeek-V2-Chat"],
    "command-r": ["Command-R", "Command-R-plus"],
    "aya": ["Aya-23-8B", "Aya-23-35B"],
    "falcon": ["Falcon-7B", "Falcon-40B", "Falcon-180B"],
    "m2": ["M2-2-Base", "M2-100k"],
    "OLMoE": ["OLMoE-1B-7B"],
    "olmo": ["OLMo-7B", "OLMo-1.7-7B"],
    "smaug": ["Smaug-72B"],
    "tulu": ["Tulu-2-7B", "Tulu-2-70B"],
}


class HuggingFaceProvider(BenchmarkProvider):
    @property
    def name(self) -> str:
        return "huggingface"

    async def fetch_data(self, ollama_models: list[str]) -> list[dict[str, Any]]:
        logger.info(f"Fetching data from {self.name} provider")

        benchmarks: list[dict[str, Any]] = []
        ollama_base_names = {self._normalize_name(m) for m in ollama_models}

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(HF_DATASET_SERVER_URL)
                response.raise_for_status()
                data = response.json()

                rows = data.get("rows", [])
                logger.info(f"Fetched {len(rows)} rows from HuggingFace dataset")

                for row_wrapper in rows:
                    row = row_wrapper.get("row", {})
                    hf_name = row.get("model_name", "")
                    if not hf_name:
                        continue

                    matched_ollama = self._match_model(hf_name, ollama_models, ollama_base_names)
                    if not matched_ollama:
                        continue

                    # Parse the results JSON string
                    results_json = row.get("results", "{}")
                    try:
                        results = json.loads(results_json) if isinstance(results_json, str) else results_json
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse results JSON for {hf_name}")
                        # Log the actual row structure for debugging
                        logger.warning(f"Row keys: {row.keys()}")
                        logger.warning(f"Row sample: {dict(list(row.items())[:5])}")
                        results = {}

                    # Extract scores from the nested structure
                    scores = self._extract_scores(results, row)
                    
                    # Fallback: if nested extraction failed (all None), use top-level scores from row
                    if scores.get("mmlu") is None and row.get("mmlu"):
                        scores["mmlu"] = row.get("mmlu")
                        logger.info(f"Fallback: using row mmlu={scores['mmlu']} for {matched_ollama}")
                    if scores.get("humaneval") is None and row.get("humaneval"):
                        scores["humaneval"] = row.get("humaneval")
                    if scores.get("math") is None and row.get("math"):
                        scores["math"] = row.get("math")
                    if scores.get("gpqa") is None and row.get("gpqa"):
                        scores["gpqa"] = row.get("gpqa")
                    
                    logger.info(f"Scores before calculation: {scores}")
                    capability_scores = self._calculate_scores(scores)
                    logger.info(f"Capability scores for {matched_ollama}: {capability_scores}")

                    benchmarks.append({
                        "ollama_name": matched_ollama,
                        "full_name": hf_name,
                        "mmlu": scores.get("mmlu"),
                        "humaneval": scores.get("humaneval"),
                        "math": scores.get("math"),
                        "gpqa": scores.get("gpqa"),
                        "hellaswag": scores.get("hellaswag"),
                        "winogrande": scores.get("winogrande"),
                        "truthfulqa": scores.get("truthfulqa"),
                        "mmlu_pro": scores.get("mmlu_pro"),
                        **capability_scores,
                        "parameters": self._extract_parameters(hf_name),
                    })

                logger.info(f"Matched {len(benchmarks)} models from HuggingFace")
                return benchmarks

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching HuggingFace data: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Request error fetching HuggingFace data: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch HuggingFace data: {e}")
            return []

    def _extract_scores(self, results: dict[str, Any], row: dict[str, Any]) -> dict[str, float | None]:
        """Extract benchmark scores from the nested results structure.
        
        The Open LLM Leaderboard uses various benchmark categories with scores
        stored in 'acc_norm,none' or 'acc,none' fields.
        """
        scores: dict[str, float | None] = {
            "mmlu": None,
            "humaneval": None,
            "math": None,
            "gpqa": None,
            "hellaswag": None,
            "winogrande": None,
            "truthfulqa": None,
            "mmlu_pro": None,
        }

        def get_score(data: dict[str, Any], key: str) -> float | None:
            """Extract score from nested structure, preferring acc_norm over acc."""
            if not isinstance(data, dict):
                return None
            if key in data:
                val = data[key]
                if isinstance(val, (int, float)):
                    return float(val)
            # Try common leaderboard field names
            for score_key in ["acc_norm,none", "acc,none", "acc_norm", "acc", "exact_match,none", "exact_match"]:
                if score_key in data:
                    val = data[score_key]
                    if isinstance(val, (int, float)):
                        return float(val) * 100  # Convert to percentage
            return None

        # MMLU - averaged across all MMLU subcategories
        if "leaderboard" in results:
            leaderboard = results["leaderboard"]
            scores["mmlu"] = get_score(leaderboard, "mmlu")
            scores["math"] = get_score(leaderboard, "math")
            scores["gpqa"] = get_score(leaderboard, "gpqa")
            scores["humaneval"] = get_score(leaderboard, "humaneval")

        # MMLU Pro
        if "leaderboard_mmlu_pro" in results:
            scores["mmlu_pro"] = get_score(results["leaderboard_mmlu_pro"], "mmlu_pro")
        elif "mmlu_pro" in results:
            scores["mmlu_pro"] = get_score(results["mmlu_pro"], "mmlu_pro")

        # HellaSwag
        if "leaderboard_hellaswag" in results:
            scores["hellaswag"] = get_score(results["leaderboard_hellaswag"], "hellaswag")
        elif "hellaswag" in results:
            scores["hellaswag"] = get_score(results["hellaswag"], "hellaswag")

        # WinoGrande
        if "leaderboard_winogrande" in results:
            scores["winogrande"] = get_score(results["leaderboard_winogrande"], "winogrande")
        elif "winogrande" in results:
            scores["winogrande"] = get_score(results["winogrande"], "winogrande")

        # TruthfulQA
        if "leaderboard_truthfulqa" in results:
            scores["truthfulqa"] = get_score(results["leaderboard_truthfulqa"], "truthfulqa")
        elif "truthfulqa" in results:
            scores["truthfulqa"] = get_score(results["truthfulqa"], "truthfulqa")

        # BBH (Big Bench Hard) - try various task-specific keys
        bbh_score = None
        for key in results:
            if key.startswith("leaderboard_bbh_") or key.startswith("bbh_"):
                task_score = get_score(results[key], key)
                if task_score is not None:
                    if bbh_score is None:
                        bbh_score = []
                    bbh_score.append(task_score)
        
        # If we couldn't find specific scores, try to infer from available data
        if scores["mmlu"] is None and "all" in results:
            scores["mmlu"] = get_score(results["all"], "mmlu")

        return scores

    def _normalize_name(self, name: str) -> str:
        name = name.split(":")[0].lower()
        name = re.sub(r"[^a-z0-9]", "", name)
        return name

    def _match_model(self, hf_name: str, ollama_models: list[str], ollama_base_names: set[str]) -> str | None:
        hf_normalized = self._normalize_name(hf_name)

        # Strategy 1: Mapping table
        for ollama_base in ollama_base_names:
            for mapping_name, hf_variants in OLLAMA_MODEL_MAPPING.items():
                if mapping_name in hf_normalized or any(v.lower().replace("-", "").replace("_", "") in hf_normalized for v in hf_variants):
                    if mapping_name in ollama_base:
                        return ollama_base

        # Strategy 2: Fuzzy match
        for ollama_model in ollama_models:
            ollama_base = self._normalize_name(ollama_model)
            if hf_normalized in ollama_base or ollama_base in hf_normalized:
                return ollama_base

        return None

    def _calculate_scores(self, data: dict[str, Any]) -> dict[str, float]:
        logger.debug(f"_calculate_scores input: {data}")
        
        reasoning = 0.0
        coding = 0.0
        general = 0.0
        count = 0

        def get_score(key: str) -> float:
            val = data.get(key)
            if val is not None:
                return float(val) / 100.0
            return 0.0

        if data.get("mmlu"):
            s = get_score("mmlu")
            reasoning += s
            general += s
            count += 2

        if data.get("mmlu_pro"):
            s = get_score("mmlu_pro")
            reasoning += s * 0.8
            general += s
            count += 2

        if data.get("gpqa"):
            reasoning += get_score("gpqa")
            count += 1

        if data.get("math"):
            s = get_score("math")
            reasoning += s
            general += s * 0.5
            count += 2

        if data.get("humaneval"):
            s = get_score("humaneval")
            coding += s
            general += s
            count += 2

        if data.get("hellaswag"):
            general += get_score("hellaswag")
            count += 1

        return {
            "reasoning_score": round(reasoning / max(count, 1), 3),
            "coding_score": round(coding / max(1, (1 if data.get("humaneval") else 0) + 1), 3),
            "general_score": round(general / max(count, 1), 3),
        }

    def _extract_parameters(self, name: str) -> str | None:
        match = re.search(r"(\d+)b", name, re.IGNORECASE)
        return match.group(1).upper() + "B" if match else None
