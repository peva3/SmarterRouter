import logging

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ROUTER_",
        extra="ignore",
    )

    # Provider selection
    provider: str = Field(default="ollama")

    # Ollama settings
    ollama_url: str = Field(default="http://localhost:11434")

    # llama.cpp / llama-swap settings
    llama_cpp_url: str | None = Field(default=None)

    # OpenAI-compatible settings (for OpenAI, Anthropic, local AI, LiteLLM, etc.)
    openai_base_url: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    model_prefix: str = Field(default="")  # prepend to model names

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=11436)

    signature_enabled: bool = Field(default=True)
    signature_format: str = Field(default="\nModel: {model}")

    polling_interval: int = Field(default=60)
    profile_timeout: int = Field(default=30)
    generation_timeout: int = Field(default=120)  # Timeout for model generation (larger models need more time)
    profile_prompts_per_category: int = Field(default=3)

    router_model: str | None = Field(default=None)
    router_temperature: float = Field(default=0.0)
    router_max_tokens: int = Field(default=50)

    # Scoring weights
    prefer_smaller_models: bool = Field(default=True)
    prefer_newer_models: bool = Field(default=True)

    # Benchmark sources (comma separated)
    benchmark_sources: str = Field(default="huggingface,lmsys")

    log_level: str = Field(default="INFO")

    database_url: str = Field(default="sqlite:///router.db")

    pinned_model: str | None = Field(default=None) # Model to keep loaded in VRAM

    # Security settings
    admin_api_key: str | None = Field(default=None)  # API key for admin endpoints (if not set, admin endpoints are open)
    rate_limit_enabled: bool = Field(default=False)  # Enable rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)  # Requests per minute limit
    rate_limit_admin_requests_per_minute: int = Field(default=10)  # Admin endpoint rate limit

    @model_validator(mode="before")
    @classmethod
    def parse_log_level(cls, values: dict) -> dict:
        log_level = values.get("log_level")
        if log_level is None:
            return values
        if isinstance(log_level, int):
            for name, level in logging._levelToName.items():
                if level == log_level:
                    values["log_level"] = name
                    break
        return values


settings = Settings()


def init_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
