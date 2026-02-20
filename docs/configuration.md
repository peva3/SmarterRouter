# Configuration Reference

SmarterRouter is configured via environment variables in the `.env` file. This reference documents all available options.

## Table of Contents
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
Enable VRAM monitoring via `nvidia-smi`.

**Default:** `true`

### `ROUTER_VRAM_MONITOR_INTERVAL`
VRAM sampling interval (seconds).

**Default:** `30`

### `ROUTER_VRAM_MAX_TOTAL_GB`
Maximum VRAM the router can allocate. Leave empty to auto-detect 90% of GPU total.

**Example:** For 24GB GPU, set to `22.0` to reserve 2GB for system

**Default:** (auto-detect 90% of GPU total)

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

---

## Monitoring & Logging

### `ROUTER_LOG_LEVEL`
Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Default:** `INFO`

### `ROUTER_LOG_FORMAT`
Log format: `text` (human-readable) or `json` (structured for log aggregation)

**Default:** `text`

**For production:** Use `json` for easy parsing by log aggregation tools

### `ROUTER_POLLING_INTERVAL`
How often to check for new models in backend (seconds).

**Default:** `60`

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

# Logging
ROUTER_LOG_LEVEL=INFO
ROUTER_LOG_FORMAT=json

# Database
ROUTER_DATABASE_URL=sqlite:///data/router.db
```

See `ENV_DEFAULT` for the complete list with inline comments.
