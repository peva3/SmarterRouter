import logging
from typing import Any

from router.providers.base import BenchmarkProvider

logger = logging.getLogger(__name__)


class ArtificialAnalysisProvider(BenchmarkProvider):
    @property
    def name(self) -> str:
        return "artificial_analysis"

    async def fetch_data(self, ollama_models: list[str]) -> list[dict[str, Any]]:
        # Placeholder for now as it requires an API key
        # In a real scenario, we would fetch from https://artificialanalysis.ai/api/v2/
        logger.info(f"Fetching data from {self.name} provider (Placeholder)")
        return []
