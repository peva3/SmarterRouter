import csv
import logging
import pickle
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import httpx
import pandas as pd

from router.providers.base import BenchmarkProvider
from router.providers.huggingface import OLLAMA_MODEL_MAPPING

logger = logging.getLogger(__name__)

# URL for the raw pickle file from LMSYS Space (update date dynamically if possible, hardcoded for stability now)
LMSYS_ELO_URL = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/elo_results_20240629.pkl"
# Fallback to a known CSV or scraping if pickle fails
LMSYS_ARENA_HARD_URL = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/arena_hard_auto_leaderboard_v0.1.csv"


class LMSYSProvider(BenchmarkProvider):
    @property
    def name(self) -> str:
        return "lmsys"

    async def fetch_data(self, ollama_models: list[str]) -> list[dict[str, Any]]:
        logger.info(f"Fetching data from {self.name} provider")
        
        benchmarks: list[dict[str, Any]] = []
        
        try:
            # Try fetching Arena Hard CSV first (easier to parse)
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(LMSYS_ARENA_HARD_URL)
                if response.status_code == 200:
                    df = pd.read_csv(BytesIO(response.content))
                    return self._process_dataframe(df, ollama_models)
                
                logger.warning(f"Failed to fetch LMSYS CSV: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch LMSYS data: {e}")
            return []

    def _process_dataframe(self, df: pd.DataFrame, ollama_models: list[str]) -> list[dict[str, Any]]:
        benchmarks = []
        ollama_base_names = {self._normalize_name(m) for m in ollama_models}

        for _, row in df.iterrows():
            model_key = row.get("model") or row.get("Model")
            if not model_key:
                continue

            matched_ollama = self._match_model(str(model_key), ollama_models, ollama_base_names)
            if not matched_ollama:
                continue

            # Extract Elo (Arena Hard often correlates or contains it)
            # If standard Elo isn't in this specific CSV, we use the score as a proxy or fetch the pickle
            elo = row.get("elo") or row.get("rating") or row.get("score")
            
            if elo:
                benchmarks.append({
                    "ollama_name": matched_ollama,
                    "elo_rating": float(elo)
                })

        return benchmarks

    def _normalize_name(self, name: str) -> str:
        name = name.split(":")[0].lower()
        import re
        name = re.sub(r"[^a-z0-9]", "", name)
        return name

    def _match_model(self, lmsys_name: str, ollama_models: list[str], ollama_base_names: set[str]) -> str | None:
        norm_name = self._normalize_name(lmsys_name)
        
        # Strategy 1: Mapping table
        for ollama_base in ollama_base_names:
            for mapping_name, hf_variants in OLLAMA_MODEL_MAPPING.items():
                if mapping_name in norm_name or any(v.lower().replace("-", "").replace("_", "") in norm_name for v in hf_variants):
                    if mapping_name in ollama_base:
                        return ollama_base

        # Strategy 2: Fuzzy match
        for ollama_model in ollama_models:
            ollama_base = self._normalize_name(ollama_model)
            if norm_name in ollama_base or ollama_base in norm_name:
                return ollama_base
                
        return None
