from abc import ABC, abstractmethod
from typing import Any


class BenchmarkProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g., 'huggingface', 'lmsys')"""
        pass

    @abstractmethod
    async def fetch_data(self, ollama_models: list[str]) -> list[dict[str, Any]]:
        """
        Fetch data from source and map to Ollama models.
        Returns a list of dictionaries with keys mapping to ModelBenchmark columns.
        """
        pass
