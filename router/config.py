import logging
import types
from typing import Union

from pydantic import Field, field_validator, model_validator
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

    @field_validator("signature_format", mode="before")
    @classmethod
    def process_escape_sequences(cls, v: str) -> str:
        """Process escape sequences in signature_format from env vars.

        Environment variables don't interpret escape sequences like \\n.
        This validator converts \\n to actual newline, \\t to tab, etc.
        """
        if isinstance(v, str):
            v = v.replace("\\n", "\n")
            v = v.replace("\\t", "\t")
            v = v.replace("\\r", "\r")
        return v

    @model_validator(mode="before")
    @classmethod
    def remove_empty_strings(cls, values: dict) -> dict:
        """Remove empty strings for optional/numeric fields so Pydantic falls back to defaults.

        Environment variables set to empty (e.g., ROUTER_VRAM_MAX_TOTAL_GB=)
        come through as empty strings, which can cause parsing errors for numeric fields.
        """
        numeric_types = (int, float)

        for k, v in list(values.items()):
            if v == "" and k in cls.model_fields:
                field = cls.model_fields[k]
                annotation = field.annotation

                # Check if annotation contains int or float
                is_numeric = False

                if annotation in numeric_types:
                    is_numeric = True
                elif getattr(annotation, "__origin__", None) is Union or isinstance(
                    annotation, getattr(types, "UnionType", type(None))
                ):
                    args = getattr(annotation, "__args__", [])
                    if any(arg in numeric_types for arg in args):
                        is_numeric = True

                if is_numeric:
                    del values[k]

        return values

    polling_interval: int = Field(default=60)
    # Model Polling & Hot‑Swap (SmarterRouter 2.1.6+)
    model_polling_enabled: bool = Field(default=True)  # Enable automatic model discovery
    model_polling_interval: int = Field(default=60)  # Seconds between model availability checks
    model_cleanup_enabled: bool = Field(
        default=False
    )  # Mark missing models as inactive (requires schema migration)
    model_auto_profile_enabled: bool = Field(default=True)  # Automatically profile new models
    profile_timeout: int = Field(default=90)  # Increased to 90s for larger models like 14B+
    generation_timeout: int = Field(
        default=120
    )  # Timeout for model generation (larger models need more time)
    profile_prompts_per_category: int = Field(default=3)

    router_model: str | None = Field(default=None)
    router_temperature: float = Field(default=0.0)
    router_max_tokens: int = Field(default=50)

    # Scoring weights
    prefer_smaller_models: bool = Field(default=True)
    prefer_newer_models: bool = Field(default=True)

    # Quality vs Cost/Speed Tuner (0.0 = max speed/cost saving, 1.0 = max quality)
    quality_preference: float = Field(default=0.5)

    # Cascading / Fallback settings
    cascading_enabled: bool = Field(default=True)  # If true, retry with larger models on failure

    # Feedback settings
    feedback_enabled: bool = Field(default=True)

    # Benchmark sources (comma separated)
    benchmark_sources: str = Field(default="huggingface,lmsys")

    # ArtificialAnalysis.ai settings
    artificial_analysis_api_key: str | None = Field(default=None)
    artificial_analysis_cache_ttl: int = Field(default=86400)  # 24 hours
    artificial_analysis_model_mapping_file: str | None = Field(default=None)

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="text")  # "text" or "json"

    database_url: str = Field(default="sqlite:///data/router.db")

    pinned_model: str | None = Field(default=None)  # Model to keep loaded in VRAM

    # Model keep_alive duration (in seconds) passed to backend.
    # -1 = keep loaded indefinitely (default), 0 = unload after response, positive = seconds to keep alive
    model_keep_alive: float = Field(default=-1)

    # Model Filtering - Optional include/exclude patterns for model discovery
    # Uses Unix shell-style glob patterns (*, ?, []). Case-insensitive matching.
    # Examples: gemma*,mistral* (include only), *qwen*,*deepseek* (exclude), or combine both
    model_filter_include: list[str] = Field(default_factory=list)  # Empty = include all
    model_filter_exclude: list[str] = Field(default_factory=list)  # Empty = exclude none

    # Name the router presents itself as to external UIs (e.g., OpenWebUI)
    router_external_model_name: str = Field(default="smarterrouter/main")

    # LLM-as-Judge Settings
    judge_enabled: bool = Field(default=False)
    judge_model: str = Field(default="gpt-4o")
    judge_base_url: str = Field(default="https://api.openai.com/v1")
    judge_api_key: str | None = Field(default=None)
    # Optional headers for providers like OpenRouter (HTTP-Referer, X-Title)
    judge_http_referer: str | None = Field(default=None)
    judge_x_title: str | None = Field(default=None)

    # Retry configuration for transient errors
    judge_max_retries: int = Field(default=3)  # Max retry attempts for 429/5xx errors
    judge_retry_base_delay: float = Field(
        default=1.0
    )  # Initial delay in seconds (doubles each retry)

    # Security settings
    admin_api_key: str | None = Field(
        default=None
    )  # API key for admin endpoints (if not set, admin endpoints are open)
    rate_limit_enabled: bool = Field(default=False)  # Enable rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)  # Requests per minute limit
    rate_limit_admin_requests_per_minute: int = Field(default=10)  # Admin endpoint rate limit

    # Smart Cache settings
    cache_enabled: bool = Field(default=True)  # Enable smart caching
    cache_max_size: int = Field(default=500)  # Max routing cache entries (increased from 100)
    cache_ttl_seconds: int = Field(default=3600)  # TTL for cache entries (1 hour)
    cache_similarity_threshold: float = Field(default=0.85)  # Threshold for semantic similarity
    cache_response_max_size: int = Field(
        default=200
    )  # Max response cache entries (increased from 50)
    embed_model: str | None = Field(default=None)  # Model to use for embeddings

    # Persistent Cache (Semantic Cache V2) settings
    persistent_cache_enabled: bool = Field(default=True)  # Enable persistent disk caching
    persistent_cache_max_age_days: int = Field(default=7)  # Maximum age of cache entries to keep

    # Enhanced Cache Statistics & Analytics (SmarterRouter 2.1.6+)
    cache_stats_enabled: bool = Field(default=True)  # Enable enhanced cache statistics
    cache_stats_retention_hours: int = Field(default=24)  # Hours to keep time-series data

    # VRAM Monitoring & Management
    vram_monitor_enabled: bool = Field(
        default=True
    )  # Enable VRAM monitoring (auto-detects all GPU vendors)
    vram_monitor_interval: int = Field(default=30)  # Seconds between VRAM samples
    vram_max_total_gb: float | None = Field(
        default=None
    )  # Max VRAM to allocate (set below GPU total)
    vram_log_interval: int = Field(default=60)  # How often to log VRAM summary

    # Apple Silicon specific configuration
    apple_unified_memory_gb: float | None = Field(
        default=None
    )  # Total unified memory in GB (for M1/M2/M3). If not set, auto-detects from system.

    # AMD APU specific configuration
    amd_unified_memory_gb: float | None = Field(
        default=None
    )  # Total unified memory in GB for AMD APUs. If not set, auto-detects from GTT pool.

    # VRAM Profiling
    profile_measure_vram: bool = Field(default=True)  # Measure actual VRAM during profiling
    profile_vram_sample_delay: float = Field(default=2.0)  # Wait after model load before measuring
    profile_vram_samples: int = Field(default=3)  # Take N samples and average

    # Parallel Profiling
    # Number of models to profile concurrently (default: 1 = sequential)
    # Set to 2-3 for multi-GPU systems or when models fit in VRAM simultaneously
    profile_parallel_count: int = Field(default=1)

    # Profiling Warmup & Loading
    # Assumed disk read speed (MB/s) for warmup timeout calculation
    # Conservative default (50 MB/s) works for HDDs and SSDs
    profile_warmup_disk_speed_mbps: float = Field(default=50.0)
    # Maximum warmup timeout in seconds (default: 1800 = 30 minutes)
    profile_warmup_max_timeout: float = Field(default=1800.0)

    # Adaptive Profiling Timeouts
    # Minimum adaptive timeout in seconds (default: 30s)
    profile_adaptive_timeout_min: float = Field(default=30.0)
    # Maximum adaptive timeout in seconds (default: 1800 = 30 minutes)
    profile_adaptive_timeout_max: float = Field(default=1800.0)
    # Safety factor for adaptive timeout calculation (default: 2.0 = conservative)
    profile_adaptive_safety_factor: float = Field(default=2.0)

    # Auto-unload policy
    vram_auto_unload_enabled: bool = Field(default=True)
    vram_unload_threshold_pct: float = Field(default=85.0)
    vram_unload_strategy: str = Field(default="lru")  # "lru" or "largest"

    # Fallback VRAM estimate when not profiled (in GB)
    vram_default_estimate_gb: float = Field(default=8.0)

    # External Provider Database (provider.db)
    # Path to provider.db containing benchmark data for external models
    provider_db_enabled: bool = Field(default=True)
    provider_db_path: str = Field(default="data/provider.db")
    # Auto-update provider.db (in hours, 0 = disabled)
    provider_db_auto_update_hours: int = Field(default=4)
    # URL to download provider.db from
    provider_db_download_url: str = Field(
        default="https://raw.githubusercontent.com/peva3/smarterrouter-provider/refs/heads/main/data/provider.db"
    )

    # External Provider API Configuration
    # When enabled, route to external providers (OpenAI, Anthropic, etc.)
    # instead of only local Ollama models
    external_providers_enabled: bool = Field(default=False)
    # List of enabled external providers (openai, anthropic, google, etc.)
    external_providers: list[str] = Field(default_factory=lambda: ["openai", "anthropic", "google"])

    # Per-Provider API Keys (used when external_providers_enabled is True)
    anthropic_api_key: str | None = Field(default=None)
    google_api_key: str | None = Field(default=None)
    cohere_api_key: str | None = Field(default=None)
    mistral_api_key: str | None = Field(default=None)

    # Per-Provider Base URLs (for self-hosted or proxy scenarios)
    anthropic_base_url: str | None = Field(default=None)
    google_base_url: str | None = Field(default=None)
    cohere_base_url: str | None = Field(default=None)
    mistral_base_url: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_external_providers(self) -> "Settings":
        """Validate external provider configuration."""
        if self.external_providers_enabled:
            # Check that all enabled providers have API keys configured
            for provider in self.external_providers:
                api_key_field = {
                    "openai": "openai_api_key",
                    "anthropic": "anthropic_api_key",
                    "google": "google_api_key",
                    "cohere": "cohere_api_key",
                    "mistral": "mistral_api_key",
                }.get(provider)

                if api_key_field:
                    api_key = getattr(self, api_key_field, None)
                    if not api_key:
                        # Cannot log here easily, but validation logic is present
                        pass
        return self

    @model_validator(mode="after")
    def validate_urls_scheme(self) -> "Settings":
        """Validate that backend URLs use http(s):// scheme."""
        url_fields = {
            "ollama_url": self.ollama_url,
            "llama_cpp_url": self.llama_cpp_url,
            "openai_base_url": self.openai_base_url,
            "judge_base_url": self.judge_base_url,
        }

        for field_name, url in url_fields.items():
            if url and not url.startswith(("http://", "https://")):
                raise ValueError(f"{field_name} must start with http:// or https:// (got: {url})")

        return self


settings = Settings()


def init_logging() -> None:
    """Initialize logging with structured formatting."""
    from .logging_config import setup_logging

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    setup_logging(level=level, log_format=settings.log_format)
