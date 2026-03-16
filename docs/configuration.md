# Configuration Reference

SmarterRouter is configured via environment variables in the `.env` file. This reference documents all available options.

## Table of Contents
- [Benchmark Data Sources](#benchmark-data-sources)
- [Backend Provider Configuration](#backend-provider-configuration)
- [Security Settings](#security-settings)
- [Routing Configuration](#routing-configuration)
- [Timeout Settings](#timeout-settings)
- [Profiling Settings](#profiling-settings)
- [Cache Configuration](#cache-configuration)
- [VRAM Monitoring](#vram-monitoring)
- [Monitoring & Logging](#monitoring--logging)
- [Database](#database)
- [LLM-as-Judge](#llm-as-judge)

## Benchmark Data Sources

### `ROUTER_BENCHMARK_SOURCES`
Comma-separated list of benchmark data sources. Options:
- `huggingface` (default)
- `lmsys`
- `artificial_analysis`

Example: `ROUTER_BENCHMARK_SOURCES=huggingface,lmsys,artificial_analysis`

**Note:** Sources are queried in the order listed. If multiple sources provide data for the same model, the **last source's data wins** (non-null values overwrite earlier ones).

### `ROUTER_ARTIFICIAL_ANALYSIS_API_KEY`
API key for ArtificialAnalysis.ai (required if `artificial_analysis` in `ROUTER_BENCHMARK_SOURCES`).

Get your free API key from: <https://artificialanalysis.ai/insights>

Rate limit: 1,000 requests per day (free tier). Data is cached for 24 hours by default to stay within limits.

### `ROUTER_ARTIFICIAL_ANALYSIS_CACHE_TTL`
Cache TTL for ArtificialAnalysis data (seconds). Default: `86400` (24 hours).

Increase if you have a paid plan with higher rate limits; decrease if you need fresher data.

### `ROUTER_ARTIFICIAL_ANALYSIS_MODEL_MAPPING_FILE`
Path to YAML file mapping ArtificialAnalysis model identifiers to SmarterRouter model names.

ArtificialAnalysis uses different naming conventions than Ollama. This file lets you explicitly map their model IDs or names to your local model tags.

**Example mapping file format** (see `artificial_analysis_models.example.yaml`):

```yaml
mappings:
  # By ArtificialAnalysis model ID (UUID) - most reliable
  "2dad8957-4c16-4e74-bf2d-8b21514e0ae9": "openai/o3-mini"

  # By ArtificialAnalysis model name/slug
  "o3-mini": "openai/o3-mini"
  "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet"
  "gemini-2.5-pro": "google/gemini-2.5-pro"
```

If no explicit mapping is found, the provider attempts to auto-generate a name using the pattern `{creator-slug}/{model-slug}`.

**Why mapping needed:** Your Ollama model tags might be `llama3.1:70b` while ArtificialAnalysis calls it "Llama-3.1-70B". The mapping bridges this gap.

---

## Backend Provider Configuration

### `ROUTER_PROVIDER`
Which backend to use. Options:
- `ollama` (default) - Local Ollama instance
- `llama.cpp` - llama.cpp server
- `openai` - OpenAI-compatible API

### `ROUTER_OLLAMA_URL`
URL of your Ollama instance or OpenAI-compatible endpoint.

**Default:** `http://localhost:11434`

**Docker note:** When SmarterRouter runs in Docker and Ollama on the host, use `http://172.17.0.1:11434`.

### `ROUTER_MODEL_PREFIX`
String to prepend to all model names sent to the backend.

**Example:** `ROUTER_MODEL_PREFIX=myorg/` makes model `llama3` become `myorg/llama3`

**Use cases:** Organizational naming, model registries, API gateways.

### OpenAI-Compatible Settings
When `ROUTER_PROVIDER=openai`:

```env
ROUTER_OPENAI_BASE_URL=https://api.openai.com/v1
ROUTER_OPENAI_API_KEY=your-api-key-here
```

Works with OpenAI, Anthropic (via compatibility layer), vLLM, TGI, LiteLLM, or any OpenAI-compatible API.

---

## Security Settings

### `ROUTER_ADMIN_API_KEY` ⚠️ REQUIRED FOR PRODUCTION
Authentication key for admin endpoints (`/admin/*`). 

**⚠️ SECURITY WARNING:** Leaving this empty makes admin endpoints publicly accessible, exposing:
- Full model performance profiles
- VRAM monitoring data
- Cache management
- Reprofile controls

**Generate a secure key:**
```bash
openssl rand -hex 32
# Copy output to .env: ROUTER_ADMIN_API_KEY=sk-smarterrouter-<output>
```

**Default:** (empty - **insecure**)

### `ROUTER_RATE_LIMIT_ENABLED`
Enable rate limiting to prevent abuse and DoS attacks.

**Default:** `false`

### `ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE`
General endpoint rate limit per client IP.

**Default:** `60`

### `ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE`
Dedicated chat endpoint (`/v1/chat/completions`) rate limit per client IP.

This limit is applied specifically to chat completions and takes precedence over the general per-minute limit for that endpoint.

**Default:** `100`

### `ROUTER_RATE_LIMIT_ADMIN_REQUESTS_PER_MINUTE`
Admin endpoint rate limit per client IP.

**Default:** `10`

---

## Routing Configuration

### `ROUTER_QUALITY_PREFERENCE`
Quality vs speed tradeoff. Range: `0.0` (max speed) to `1.0` (max quality).

**Default:** `0.5`

**Effects:**
- Low (0.0-0.3): Prefers smaller, faster models
- Medium (0.4-0.6): Balanced approach
- High (0.7-1.0): Prefers larger, higher-quality models

### `ROUTER_CASCADING_ENABLED`
If a selected model fails, automatically retry with the next best capable model.

**Default:** `true`

### `ROUTER_FEEDBACK_ENABLED`
Collect user feedback (`/v1/feedback`) to improve future routing decisions.

**Default:** `true`

### `ROUTER_PREFER_SMALLER_MODELS`
Prefer smaller models for simple tasks when quality is equal.

**Default:** `true`

### `ROUTER_PREFER_NEWER_MODELS`
Prefer newer models when scores are similar.

**Default:** `true`

### `ROUTER_EXTERNAL_MODEL_NAME`
Name the router presents itself as to external UIs (e.g., OpenWebUI).

**Default:** `smarterrouter/main`

---

## Timeout Settings

### `ROUTER_GENERATION_TIMEOUT`
Timeout for model generation requests (seconds).

**Default:** `120`

**Increase for:** Large models (14B+), complex reasoning tasks

### `ROUTER_REQUEST_TIMEOUT_ENABLED`
Enable global request timeout enforcement across full request processing (routing, model loading, generation, and post-processing).

**Default:** `true`

### `ROUTER_REQUEST_TIMEOUT_SECONDS`
Overall request timeout budget in seconds. Requests exceeding this limit are cancelled and return HTTP 504.

**Default:** `300`

### `ROUTER_PROFILE_TIMEOUT`
Base timeout for profiling operations (seconds).

**Default:** `90`

**Increase for:** Profiling large models to avoid premature timeouts

---

## Profiling Settings

### `ROUTER_PROFILE_PROMPTS_PER_CATEGORY`
Number of test prompts per category (reasoning, coding, creativity) during profiling.

**Default:** `3`

**Higher values:** More accurate profiles, longer profiling time
**Lower values:** Faster profiling, less accuracy

### `ROUTER_PROFILE_MEASURE_VRAM`
Measure actual VRAM usage during profiling.

**Default:** `true`

### `ROUTER_PROFILE_VRAM_SAMPLE_DELAY`
Delay after loading model before measuring VRAM (seconds). Allows memory to stabilize.

**Default:** `2.0`

### `ROUTER_PROFILE_VRAM_SAMPLES`
Number of VRAM samples to take during profiling (averaged).

**Default:** `3`

### `ROUTER_PROFILE_ADAPTIVE_SAFETY_FACTOR`
Safety factor for adaptive timeout calculation (default: 2.0 = conservative). Higher = more buffer, lower = more aggressive.

**Default:** `2.0`

---

## Cache Configuration

### `ROUTER_CACHE_ENABLED`
Enable smart caching of routing decisions and responses.

**Default:** `true`

### `ROUTER_CACHE_MAX_SIZE`
Maximum number of routing cache entries (SHA-256 hash based).

**Default:** `500`

### `ROUTER_CACHE_TTL_SECONDS`
Time-to-live for cache entries (seconds).

**Default:** `3600` (1 hour)

### `ROUTER_CACHE_BACKEND`
Cache backend implementation.

- `memory` (default)
- `redis`

### `ROUTER_REDIS_URL`
Redis connection URL used when `ROUTER_CACHE_BACKEND=redis`.

**Default:** `redis://localhost:6379`

### `ROUTER_REDIS_CACHE_PREFIX`
Prefix for Redis cache keys.

**Default:** `smarterrouter:`

### `ROUTER_CACHE_CLEANUP_INTERVAL_HOURS`
Interval for background persistent-cache cleanup task.

Set to `0` to disable the periodic cleanup task.

**Default:** `24`

### `ROUTER_CACHE_RESPONSE_MAX_SIZE`
Maximum number of response cache entries.

**Default:** `200`

### `ROUTER_EMBED_MODEL`
Embedding model for semantic similarity matching. If set, enables semantic caching in addition to exact hash matching.

**Example:** `nomic-embed-text:latest`

### `ROUTER_CACHE_SIMILARITY_THRESHOLD`
Similarity threshold for semantic matching (0.0-1.0). Higher = more strict matching.

**Default:** `0.85`

---

## VRAM Monitoring

### `ROUTER_VRAM_MONITOR_ENABLED`
Enable VRAM monitoring with auto-detection across all GPU vendors (NVIDIA, AMD, Intel, Apple Silicon).

**Default:** `true`

### `ROUTER_APPLE_UNIFIED_MEMORY_GB`
Override auto-detected unified memory for Apple Silicon Macs. SmarterRouter estimates GPU memory as a percentage of system RAM (default: 75%). Set this to explicitly define the total GB available for GPU workloads on Apple Silicon.

**Default:** (auto-detect as 75% of system RAM)

### `ROUTER_VRAM_MONITOR_INTERVAL`
VRAM sampling interval (seconds).

**Default:** `30`

### `ROUTER_VRAM_MAX_TOTAL_GB`
Maximum VRAM the router can allocate. Leave empty to auto-detect 90% of total GPU memory across all detected GPUs.

**Example:** For 24GB GPU, set to `22.0` to reserve 2GB for system

**Default:** (auto-detect 90% of total detected VRAM)

### `ROUTER_VRAM_UNLOAD_THRESHOLD_PCT`
VRAM utilization percentage for warnings (not automatic unloads).

**Default:** `85.0`

### `ROUTER_VRAM_AUTO_UNLOAD_ENABLED`
Automatically unload unused models when VRAM pressure is high.

**Default:** `true`

### `ROUTER_VRAM_UNLOAD_STRATEGY`
Strategy for selecting models to unload:
- `lru` (default) - least recently used
- `largest` - unload biggest models first

### `ROUTER_VRAM_DEFAULT_ESTIMATE_GB`
Default VRAM estimate for models without measured data.

**Default:** `8.0`

### `ROUTER_MODEL_KEEP_ALIVE`
Controls how long models stay loaded in VRAM after each request (passed to backend's `keep_alive` parameter).

- `-1` (default): Keep models loaded indefinitely. They stay in VRAM until explicitly unloaded or the router shuts down.
- `0`: Unload models immediately after each response. Good for conserving VRAM at the cost of slower subsequent requests (model must reload).
- Positive integer: Number of seconds to keep the model loaded after the response (e.g., `300` = 5 minutes).

**Note:** This setting only affects backends that support `keep_alive` (Ollama). Other backends may ignore it.

**Example:** Set `ROUTER_MODEL_KEEP_ALIVE=0` to ensure only the most recently used model remains loaded, freeing VRAM for other applications.

### `ROUTER_MODEL_FILTER_INCLUDE`

Comma-separated list of glob patterns to include when discovering models. Only models matching these patterns will be available for routing and profiling. Case-insensitive matching.

**Patterns:**
- `*` matches everything
- `?` matches any single character
- `[seq]` matches any character in seq
- `[!seq]` matches any character not in seq

**Default:** (empty - include all models)

**Examples:**
- `ROUTER_MODEL_FILTER_INCLUDE=gemma*,mistral*` - Only include gemma and mistral models
- `ROUTER_MODEL_FILTER_INCLUDE=llama*,phi*` - Include llama and phi model families

### `ROUTER_MODEL_FILTER_EXCLUDE`

Comma-separated list of glob patterns to exclude when discovering models. Models matching these patterns will be removed from the available set. Case-insensitive matching. Exclude patterns take precedence over include patterns.

**Default:** (empty - exclude no models)

**Examples:**
- `ROUTER_MODEL_FILTER_EXCLUDE=*qwen*,*deepseek*` - Exclude qwen and deepseek models
- `ROUTER_MODEL_FILTER_EXCLUDE=*test*,*dev*` - Exclude test/dev models

### Combining Include and Exclude

You can use both settings together. The filtering logic is:

1. First, exclude patterns are applied (models matching exclude are removed)
2. Then, include patterns are applied (if include is non-empty, only matching models remain)

**Example - Use gemma and mistral but exclude quantized versions:**
```
ROUTER_MODEL_FILTER_INCLUDE=gemma*,mistral*
ROUTER_MODEL_FILTER_EXCLUDE=*q4_*,*q5_*,*q8_*
```

**Example - Exclude everything except specific models:**
```
ROUTER_MODEL_FILTER_EXCLUDE=*
ROUTER_MODEL_FILTER_INCLUDE=llama3.1:8b,phi3:mini
```

**Multi-GPU Support:** SmarterRouter automatically detects all available GPUs regardless of vendor and combines their memory. GPU indexing is global across vendors (0, 1, 2, ...). If no GPUs are detected on startup, VRAM monitoring is disabled with a warning. GPU detection runs on every startup, so adding new hardware requires only a restart.

**Supported Vendors:**
- **NVIDIA:** via `nvidia-smi`
- **AMD:** via `rocm-smi` or sysfs
- **Intel:** Arc GPUs with dedicated VRAM (via sysfs `lmem_total`)
- **Apple Silicon:** Unified memory estimation (default 75% of system RAM)

---

## Monitoring & Logging

### `ROUTER_LOG_LEVEL`
Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Default:** `INFO`

### `ROUTER_LOG_FORMAT`
Log format: `text` (human-readable) or `json` (structured for log aggregation)

**Default:** `text`

**For production:** Use `json` for easy parsing by log aggregation tools

When using `json` logging, warning/error records include structured context fields where available (e.g., `request_id`, `user_ip`, `model_name`, `prompt_hash`) to improve incident triage and cross-service correlation.

### `ROUTER_POLLING_INTERVAL`
How often to check for new models in backend (seconds).

**Default:** `60`

### `ROUTER_ENABLE_RESPONSE_COMPRESSION`
Enable gzip compression middleware for API responses.

**Default:** `false`

### `ROUTER_COMPRESSION_MINIMUM_SIZE`
Minimum response size (bytes) before gzip compression is applied.

**Default:** `1024`

### `ROUTER_ENABLE_SLOW_QUERY_LOGGING`
Enable slow-request logging middleware.

When enabled, requests that exceed `ROUTER_SLOW_QUERY_THRESHOLD_MS` are logged with request metadata and a stack snapshot.

**Default:** `false`

### `ROUTER_SLOW_QUERY_THRESHOLD_MS`
Slow request threshold in milliseconds.

**Default:** `500`

---

## Database

### `ROUTER_DATABASE_URL`
Database connection URL.

**Default:** `sqlite:///data/router.db`

**For PostgreSQL in production:**
```
postgresql://user:password@localhost:5432/smarterrouter
```

**Note:** The database file and parent directories are automatically created on startup.

### Connection Pooling

These settings tune SQLAlchemy connection pooling (primarily for non-SQLite backends):

- `ROUTER_DATABASE_POOL_SIZE` (default: `10`)
- `ROUTER_DATABASE_MAX_OVERFLOW` (default: `20`)
- `ROUTER_DATABASE_POOL_RECYCLE` (default: `3600`)
- `ROUTER_DATABASE_POOL_PRE_PING` (default: `true`)

For SQLite, pool settings are less impactful due to file-based locking, but are still accepted.

### provider.db Reliability Controls

- `ROUTER_PROVIDER_DB_ENABLED` - Enable provider.db benchmark usage (**default:** `true`)
- `ROUTER_PROVIDER_DB_PATH` - Path to provider.db (**default:** `data/provider.db`)
- `ROUTER_PROVIDER_DB_MAX_AGE_HOURS` - Mark provider.db stale if `last_build` is older than this many hours (**default:** `168`)
- `ROUTER_PROVIDER_DB_AUTO_UPDATE_HOURS` - Background auto-update interval (**default:** `4`)
- `ROUTER_PROVIDER_DB_DOWNLOAD_URL` - Download source URL for provider.db

#### DB slowness fallback

- `ROUTER_DB_SLOW_FALLBACK_ENABLED` - Enable temporary stale-cache fallback when provider.db is slow/unavailable (**default:** `true`)
- `ROUTER_DB_SLOW_QUERY_THRESHOLD_MS` - Query latency threshold that triggers degraded fallback window (**default:** `250`)
- `ROUTER_DB_SLOW_FALLBACK_WINDOW_SECONDS` - Duration of degraded fallback window after slow/failing query (**default:** `30`)
- `ROUTER_DB_STALE_CACHE_MAX_AGE_SECONDS` - Maximum age of in-memory benchmark cache allowed for fallback serving (**default:** `300`)

---

## Backend Resilience

### Retry Controls

- `ROUTER_BACKEND_RETRY_ENABLED` - Enable retry for transient backend errors (**default:** `true`)
- `ROUTER_BACKEND_MAX_RETRIES` - Maximum retry attempts (**default:** `3`)
- `ROUTER_BACKEND_RETRY_BASE_DELAY` - Initial backoff delay in seconds (**default:** `0.5`)
- `ROUTER_BACKEND_RETRY_MAX_DELAY` - Maximum backoff delay in seconds (**default:** `8.0`)

Retryable failures include timeouts, network errors, HTTP 429, and HTTP 5xx.

### Circuit Breaker Controls

- `ROUTER_BACKEND_CIRCUIT_BREAKER_ENABLED` (**default:** `true`)
- `ROUTER_BACKEND_CIRCUIT_BREAKER_FAILURE_THRESHOLD` (**default:** `5`)
- `ROUTER_BACKEND_CIRCUIT_BREAKER_RESET_TIMEOUT` (**default:** `60.0`)
- `ROUTER_BACKEND_CIRCUIT_BREAKER_HALF_OPEN_MAX_ATTEMPTS` (**default:** `3`)
- `ROUTER_BACKEND_CIRCUIT_BREAKER_SLIDING_WINDOW_SIZE` (**default:** `100`)

When enabled, backend operations open their circuit after repeated failures, fail fast while open, and probe recovery in half-open state.

---

## Dead Letter Queue (DLQ)

### `ROUTER_DLQ_ENABLED`
Enable persistent dead-letter-queue capture for failed background jobs.

**Default:** `true`

### `ROUTER_DLQ_MAX_RETRIES`
Maximum retry attempts per failed background task before marking it `dead`.

**Default:** `3`

### `ROUTER_DLQ_RETRY_BASE_DELAY_SECONDS`
Base retry delay in seconds for DLQ retries. Backoff is exponential per attempt.

**Default:** `60`

### `ROUTER_DLQ_AUTO_RETRY_BATCH_SIZE`
Maximum number of due DLQ entries retried per retry-worker iteration.

**Default:** `10`

DLQ captures failures from background sync/cleanup workflows and stores them in `background_task_dlq` for later inspection and retry.

---

## LLM-as-Judge

### `ROUTER_JUDGE_ENABLED`
Use an LLM to grade model outputs during profiling (higher quality scores).

**Default:** `false` (requires external API)

**Enable for:** More accurate model capability assessment

### `ROUTER_JUDGE_MODEL`
Model to use as the judge (e.g., `gpt-4o`, `claude-3-opus`).

**Default:** `gpt-4o`

### `ROUTER_JUDGE_BASE_URL`
Base URL for judge's API endpoint.

**Default:** `https://api.openai.com/v1`

### `ROUTER_JUDGE_API_KEY`
API key for judge's service.

**Default:** (empty)

### `ROUTER_JUDGE_HTTP_REFERER`
HTTP referer header (required by some providers like OpenRouter).

**Default:** (empty)

### `ROUTER_JUDGE_X_TITLE`
X-Title header for provider analytics.

**Default:** (empty)

### `ROUTER_JUDGE_MAX_RETRIES`
Max retry attempts for transient errors.

**Default:** `3`

### `ROUTER_JUDGE_RETRY_BASE_DELAY`
Initial retry delay in seconds (doubles on each retry).

**Default:** `1.0`

---

## Complete Example `.env` File

```env
# Backend
ROUTER_PROVIDER=ollama
ROUTER_OLLAMA_URL=http://localhost:11434

# Security (CRITICAL FOR PRODUCTION)
ROUTER_ADMIN_API_KEY=sk-smarterrouter-$(openssl rand -hex 32)
ROUTER_RATE_LIMIT_ENABLED=true

# Routing
ROUTER_QUALITY_PREFERENCE=0.5
ROUTER_PINNED_MODEL=phi3:mini
ROUTER_CASCADING_ENABLED=true

# Cache
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_MAX_SIZE=500

# VRAM
ROUTER_VRAM_MAX_TOTAL_GB=22.0
ROUTER_VRAM_AUTO_UNLOAD_ENABLED=true
ROUTER_MODEL_KEEP_ALIVE=-1

# Logging
ROUTER_LOG_LEVEL=INFO
ROUTER_LOG_FORMAT=json

# Database
ROUTER_DATABASE_URL=sqlite:///data/router.db
```

See `ENV_DEFAULT` for the complete list with inline comments.
