# SmarterRouter: Architectural Deep Dive

This document provides a technical overview of SmarterRouter, detailing the design philosophy, component interactions, and the rationale behind specific architectural choices.

---

## 1. The Core Philosophy

The primary goal of this project is to solve the "Paradox of Choice" in local LLM deployments. As the number of available models grows (Mistral, Llama, Qwen, DeepSeek, etc.), users often find it difficult to know which model is best for a specific prompt. 

Most users default to their largest model, which is slow, or their fastest model, which might be too simple for complex tasks. This router acts as an intelligent middleware that makes that decision automatically, balancing capability, speed, and resource constraints.

Additionally, the router now supports **Vector Embeddings** generation, making it a complete local AI gateway. It can handle both generative tasks (chat) and retrieval tasks (embeddings) through a single, unified API.

---

## 2. Component Architecture

### 2.1 Backend Abstraction Layer (`router/backends/`)
We didn't want to build a tool that only works with Ollama. The backend layer uses a Protocol-based abstraction (similar to a Java Interface) to ensure that the core routing logic is decoupled from the specific LLM engine.

- **Ollama Backend**: The primary target for local users. Supports VRAM management, model unloading, and native embeddings. Explicitly implements `LLMBackend` protocol.
- **llama.cpp Backend**: Designed for high-performance deployments using the standard `llama.cpp` server. Supports embeddings. Protocol-compliant.
- **OpenAI-Compatible Backend**: Allows the router to act as a bridge to external APIs (OpenAI, Anthropic, or even other instances of this router). Full protocol adherence.

**Why this matters:** It future-proofs the system. If a new high-performance engine emerges tomorrow, we only need to implement one Python class to support it. The explicit protocol inheritance provides compile-time checking and better IDE support.

### 2.2 The Routing Engine (`router/router.py`)
The "Brain" of the system. It handles the scoring and selection process using a multi-weighted algorithm.

- **Query Difficulty Prediction**: Before choosing a model, the router analyzes the prompt. It looks for logic indicators, code structures, and instruction density to decide if the task is "Easy" or "Hard."
- **Scoring Heuristics**: It combines three distinct data points:
    1. **Static Benchmarks**: External data from HuggingFace/LMSYS (how the model performs in general).
    2. **Runtime Profiles**: Local data from our profiler (how the model performs on *your* hardware).
    3. **Name Affinity**: Heuristic matching for specific tasks (e.g., routing `.py` requests to `*coder` models).
- **Quality vs. Speed Tuner**: The `ROUTER_QUALITY_PREFERENCE` setting acts as a global bias. A low value prioritizes throughput; a high value prioritizes benchmark scores and model size.
- **Category-Minimum Size**: Prevents small models from being selected for complex tasks. Each category (coding, reasoning, creativity) has minimum size requirements based on prompt complexity:
    - **Coding**: simple=0B, medium=4B+, hard=8B+
    - **Reasoning**: simple=0B, medium=4B+, hard=8B+
    - **Creativity**: simple=0B, medium=1B+, hard=4B+
    - Models below minimum get a severe penalty (-10 * size deficit).

### 2.3 The Profiling Pipeline (`router/profiler.py` & `router/judge.py`)
Model evaluation is often subjective. We moved from a basic "did it respond?" check to a sophisticated evaluation pipeline.

- **Standardized Prompts**: We use a curated set of prompts inspired by **MT-Bench**. This ensures models are tested on reasoning, coding, and creativity in a consistent way.
- **LLM-as-Judge**: This is a critical feature for high-quality deployments. If enabled, the router uses a powerful model (the "Judge") to grade the responses of smaller models. 
- **Capability Detection**: The profiler doesn't just look at scores; it probes for Vision and Tool-Calling support, ensuring requests requiring these features aren't routed to models that will fail them.

**Rationale:** Local hardware varies wildly. A model that is fast on an A100 might be unusable on a laptop. Local profiling is the only way to get an accurate "Speed" score for your specific environment.

### 2.4 Resource & VRAM Management

Running multiple models locally requires careful VRAM management. The router now includes a comprehensive VRAM monitoring and management system to prevent out-of-memory errors and optimize model usage.

**VRAM Monitoring**: A background `VRAMMonitor` task runs at configurable intervals (default 30s), sampling GPU memory using `nvidia-smi`. It maintains a rolling buffer of metrics and logs concise summaries (total, used, free, utilization) to the application log at a separate interval. The monitor can also query per-model estimated usage based on the router's allocation tracking.

**VRAM Profiling**: During model profiling, the router measures the actual VRAM footprint of each model. After loading a model, a baseline reading is taken, then after running benchmark prompts, the increase in VRAM is attributed to the model. This measured value (`vram_required_gb`) is stored in the `ModelProfile` along with a timestamp (`vram_measured_at`). This provides accurate, hardware-specific memory requirements for routing decisions.

**Admin VRAM Endpoint**: The `/admin/vram` endpoint provides real-time insight into GPU memory status. It returns:
- Current total, used, and free VRAM (GB)
- Utilization percentage
- List of loaded models and their estimated memory allocation
- Historical metrics (limited to recent entries) for trend analysis
The endpoint requires admin authentication if `ROUTER_ADMIN_API_KEY` is set.

**Simplified Configuration**: The router uses a single setting `ROUTER_VRAM_MAX_TOTAL_GB` to define the maximum VRAM budget for models. There is no separate `headroom_gb`; instead, the system automatically reserves a fixed 0.5GB internal fragmentation buffer to account for measurement errors and memory fragmentation. If `ROUTER_VRAM_MAX_TOTAL_GB` is not explicitly set, the router will auto-detect the GPU's total VRAM and default to 90% of that value, leaving 10% for the system and drivers.

**VRAM-Aware Routing**: The `VRAMManager` tracks which models are currently loaded (or pinned) and their estimated VRAM usage. Before loading a new model, it checks if the model fits within the available budget (`max_vram - buffer - sum(loaded)`). If not, it can automatically unload models according to a configurable strategy (LRU or largest-first) to make room, respecting pinned models that should never be unloaded. This coordination happens in `RouterEngine` via `vram_manager` and integrates with backends (Ollama, llama.cpp) that support model loading/unloading.

**Profiling Integration**: When profiling new models with `ROUTER_PROFILE_MEASURE_VRAM=true`, the router populates the `vram_required_gb` field automatically. If the profiler is not run, the router falls back to a default estimate (`ROUTER_VRAM_DEFAULT_ESTIMATE_GB`) for capacity planning. This ensures the router can function even without explicit measurements.

**Multi-GPU Support**: The VRAM monitor automatically detects all NVIDIA GPUs and aggregates their memory. The `total_vram_gb` used by the manager is the sum of all GPU memory. The `/admin/vram` endpoint also reports per‑GPU breakdowns, allowing you to see utilization on each device. The router does not currently pin models to specific GPUs; it relies on the backend's default device placement.

**AMD APU Unified Memory**: AMD APUs (Accelerated Processing Units) like the Ryzen AI 300 series with Radeon 800M graphics use a unified memory architecture where CPU and GPU share system RAM. The VRAM monitor handles this specially:

- **Detection**: GPUs with <4GB VRAM are detected as APUs
- **Memory Source**: Uses GTT (Graphics Translation Table) pool instead of VRAM carve-out
- **GTT vs VRAM**: 
  - `mem_info_vram_*`: Small BIOS carve-out (512MB-8GB) - NOT the usable memory
  - `mem_info_gtt_*`: Dynamic pool from system RAM - the ACTUAL usable memory
- **BIOS Configuration**: UMA Frame Buffer should be set to MINIMUM (not maximum) to avoid wasting RAM
- **Manual Override**: `ROUTER_AMD_UNIFIED_MEMORY_GB` allows setting unified memory size manually if auto-detection fails

This architecture allows APUs to use nearly all system RAM for GPU workloads, unlike discrete GPUs with fixed VRAM.

### 2.5 External Provider Integration

SmarterRouter extends beyond local models with **external provider support**, allowing you to route to cloud APIs (OpenAI, Anthropic, Google, etc.) alongside your local Ollama models.

#### provider.db: Benchmark Database for 400+ Models

External provider integration relies on **provider.db**, a SQLite database containing benchmark scores for hundreds of models from OpenRouter. It's built by the [smarterrouter-provider](https://github.com/peva3/smarterrouter-provider) project, which aggregates data from:

- **LMSYS Chatbot Arena** - ELO ratings from human preferences
- **LiveBench** - Reasoning tasks
- **BigCodeBench** - Coding ability
- **MMLU, MMLU-Pro** - General knowledge
- **GSM8K, ARC, BBH** - Math and reasoning
- **SWE-bench, HumanEval** - Code generation
- **And many more...** (28+ benchmark sources)

**Auto-Update:** provider.db is automatically downloaded every 4 hours (configurable) by the background sync task in `main.py`. The file is stored at `data/provider.db` (200KB, ~400+ models).

**Schema:**

```sql
CREATE TABLE model_benchmarks (
    model_id TEXT PRIMARY KEY,           -- e.g., "openai/gpt-4o"
    reasoning_score REAL NOT NULL,       -- 0-100 scale
    coding_score REAL NOT NULL,          -- 0-100 scale
    general_score REAL NOT NULL,         -- 0-100 scale
    elo_rating INTEGER NOT NULL,         -- 1000-2000 scale
    last_updated TIMESTAMP,
    archived INTEGER DEFAULT 0
)
```

#### Model Naming Convention

External models use **provider prefixes** to distinguish them from local models:

```
openai/gpt-4o
anthropic/claude-3-opus
google/gemini-1.5-pro
cohere/command-r-plus
mistral/mistral-large
```

The prefix (before `/`) identifies the provider and is used by the router to determine which backend to use.

#### BackendRegistry: Unified Multi-Backend Management

The **BackendRegistry** (`router/backends/registry.py`) manages all available backends and determines where to route each model:

```python
class BackendRegistry:
    def get_backend_for_model(self, model_name: str) -> tuple[str, LLMBackend | None]:
        # 1. Check if external model from provider.db
        if "/" in model_name and provider_db has benchmark:
            return ("external", None)

        # 2. Check if local model (no slash)
        if local_backend and "/" not in model_name:
            return ("local", local_backend)

        # 3. Default to local if available
        return ("local", local_backend) or ("unknown", None)
```

**Key insight:** The router uses the model name format to determine routing:
- Models with `/` are external (looked up in provider.db)
- Models without `/` are local Ollama models

This naming convention eliminates configuration - just use `openai/gpt-4o` and the rest is automatic.

#### ExternalBackendFactory: Provider-Specific Instances

When an external model is selected, the **ExternalBackendFactory** (`router/backends/external.py`) creates or retrieves an appropriate backend:

```python
PROVIDER_CONFIGS = {
    "openai": {
        "default_base_url": "https://api.openai.com/v1",
        "api_key_field": "openai_api_key",
        "base_url_field": "openai_base_url",
        "model_prefix": "",
    },
    "anthropic": {
        "default_base_url": "https://api.anthropic.com/v1",
        "api_key_field": "anthropic_api_key",
        "base_url_field": "anthropic_base_url",
        "model_prefix": "",  # Anthropic uses full model ID
    },
    "google": {
        "default_base_url": "https://generativelanguage.googleapis.com/v1",
        "api_key_field": "google_api_key",
        "base_url_field": "google_base_url",
        "model_prefix": "models/",
    },
    # ... cohere, mistral, etc.
}
```

The factory is initialized on-demand and caches backend instances per provider for efficiency.

#### Chat Request Flow (External Provider)

1. **User sends request** to SmarterRouter: `POST /v1/chat/completions` with `model: "openai/gpt-4o"`

2. **RouterEngine** selects the best model via `_keyword_dispatch()` or `_llm_dispatch()`

3. **BackendRegistry** determines this is an external model (has `/` prefix and exists in provider.db)

4. **ExternalBackendFactory** creates an `OpenAIBackend` instance for the `openai` provider (or retrieves cached instance)

5. **OpenAIBackend** transforms the request:
   - Applies model prefix (if configured) - OpenAI typically uses empty prefix
   - Adds `Authorization: Bearer <api_key>` header
   - Forwards to `https://api.openai.com/v1/chat/completions`

6. **External API** processes request and returns response

7. **OpenAIBackend** transforms response back to Ollama-compatible format and returns to RouterEngine

8. **RouterEngine** adds signature (if enabled) and returns to client

#### Configuration

Enable external providers with these environment variables:

```bash
# Enable external routing
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google

# API keys (required)
ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
ROUTER_GOOGLE_API_KEY=...
ROUTER_COHERE_API_KEY=...
ROUTER_MISTRAL_API_KEY=...

# Optional: Custom base URLs (for proxies/self-hosted)
ROUTER_ANTHROPIC_BASE_URL=https://custom-endpoint.com/v1
ROUTER_GOOGLE_BASE_URL=https://custom-endpoint.com/v1
```

All external provider settings are **optional defaults** - if you don't set them, the router will fall back to local-only mode.

#### Data Flow: External Model Selection

The router's scoring algorithm works identically for external models:

1. **Local profiles** (if model is also available locally) are loaded from `router.db`
2. **External benchmarks** are loaded from `provider.db` via `get_benchmarks_for_models_with_external()`
3. Scores are merged: local data takes precedence, external data fills gaps
4. **Combined scores** are calculated using the standard formula:
   ```
   Combined = (benchmark × 1.5 × Q) + (elo × 1.0 × Q) + (profile × 0.8 × Q)
   ```
5. Model with highest score is selected, regardless of backend type

#### Backend Compatibility: OpenAI-Compatible API

The **OpenAIBackend** (`router/backends/openai.py`) handles all external providers because they offer OpenAI-compatible endpoints:

- **OpenAI**: Native OpenAI API (`/v1/chat/completions`)
- **Anthropic**: Anthropic's native Messages API uses `/v1/messages` but they also offer an OpenAI-compatible endpoint (`/v1/chat/completions`) for basic use cases. For advanced features (prompt caching), you'd need a dedicated Anthropic backend (future enhancement).
- **Google**: Gemini API has OpenAI-compatible proxy
- **Cohere**: Cohere API is OpenAI-compatible
- **Mistral**: Mistral API is OpenAI-compatible

All use the same request/response format: JSON with `messages`, `model`, `temperature`, etc., and return `choices[0].message.content`.

#### Capability Detection

External models' capabilities (vision, tool calling) are detected via **keyword matching** in the model name, same as local models:

```python
VISION_KEYWORDS = ['llava', 'pixtral', 'vision', 'gpt-4o', 'claude-3', ...]
TOOL_KEYWORDS = ['gpt-4', 'claude-3', 'mistral-large', ...]
```

This happens in `router/router.py`'s `_has_capability()` method. For external models like `openai/gpt-4o-vision`, the router correctly identifies vision capability because `gpt-4o` is in the vision keywords list.

#### Security Considerations

**API Key Isolation:**
- Each provider has its own dedicated environment variable (no shared keys)
- Keys are never logged or exposed in model responses
- Use Docker secrets or vault solutions in production

**Network Security:**
- All provider endpoints use HTTPS (enforced by validation)
- No certificate pinning - relies on system CA store
- Consider using a proxy/VPC for additional isolation

**Least Privilege:**
- Only grant the `external_providers_enabled` permission to providers you trust
- Each enabled provider requires a valid API key
- Misconfigured providers are logged but don't crash the system

#### Performance Characteristics

**Latency:**
- Local models: ~100-500ms (depending on model size and hardware)
- External APIs: ~1000-3000ms (network + server processing)
- Cache hits: ~1-10ms (semantic cache bypasses model entirely)

**Cost:**
- Local: $0 (except electricity)
- External: $ per token (OpenAI: $2.50-30/1M tokens; Anthropic: $3-15/1M tokens)
- SmarterRouter helps reduce costs by routing simple tasks to cheaper models

**Rate Limiting:**
- External APIs have rate limits (OpenAI: varies by tier; Anthropic: 1000-10000 RPM)
- SmarterRouter does NOT implement per-provider rate limiting yet - that's a future enhancement
- If you hit limits, the router will see HTTP 429 errors and could fall back to local models

#### Limitations and Future Work

**Current Limitations:**
1. **No streaming support for external providers** - The current implementation only returns complete responses. Streaming (server-sent events) is planned for a future release.
2. **No prompt caching for Anthropic** - Anthropic's prompt caching requires using their native `/v1/messages` endpoint with special headers. The OpenAI-compatible endpoint doesn't support it.
3. **No tool calling for external providers** - Tool/function calling is supported in the OpenAI backend but not yet tested with external providers.
4. **Single fallback model** - If an external API fails, the router could try alternative models (local or other external). Currently falls back to keyword dispatch.
5. **No usage tracking per provider** - Token usage is logged but not aggregated per external provider for cost reporting.

**Planned Enhancements:**
- Streaming support for external providers
- Per-provider rate limit configuration
- Automatic fallback to alternative providers on API errors
- Cost-aware routing (factor API costs into scoring)
- Provider-specific features (Anthropic prompt caching, OpenAI vision)
- Usage analytics and cost reports

---

## 3. Data Flow: Anatomy of a Request

### 3.1 Chat Completions (Intelligent Routing)
1. **Ingress**: A user sends an OpenAI-style `/v1/chat/completions` request to the router.
2. **Analysis**:
    - The router identifies if the request needs Vision or specific Tools.
    - The difficulty predictor tags the request as Easy, Medium, or Hard.
3. **Selection**:
    - The `RouterEngine` pulls all profiled models from the database.
    - It filters out models that lack required capabilities (e.g., Vision).
    - It calculates a weighted score for each remaining model.
    - The model with the highest score is selected.
4. **Execution & Tool Loop**:
    - The router checks if the model is loaded.
    - If a different model is in VRAM, it triggers an unload.
    - It forwards the request to the backend, passing through all standard parameters (temperature, top_p, etc.).
    - **Tool Execution**: If the model response contains `tool_calls`, the router executes the specified tools (e.g., web search) and sends the results back to the model, looping up to 5 times.
5. **Egress**:
    - The response is streamed back to the user (if requested).
    - An optional signature is appended (e.g., "Model: deepseek-r1:7b").
    - Token usage is calculated and returned.
6. **Feedback (Optional)**:
    - If the user provides a rating via `/v1/feedback`, that score is saved to the database and will influence that model's selection in the future.

### 3.2 Embeddings (Direct Forwarding)
The `/v1/embeddings` endpoint works differently from chat:

1. **Ingress**: User sends an embedding request with a specific model name (e.g., `nomic-embed-text`).
2. **Validation**: The request is validated against Pydantic schemas.
3. **Execution**: The request is forwarded directly to the specified backend model.
4. **Response**: The embedding vectors are returned in OpenAI-compatible format.

*Note: The router does not currently "route" embeddings requests intelligently. Embedding models are typically specialized and specific to the use case (e.g., semantic search vs. classification), so the user is expected to select the correct model.*

---

## 4. Database & Storage (`router/models.py`)

We chose **SQLite** via **SQLAlchemy** for storage.

- **Why SQLite?** Zero configuration. It's a single file (`router.db`) that makes the router truly "plug-and-play."
- **Audit Logging**: Every routing decision and response time is logged. This allows for future "Post-Mortem" analysis to see if the router is making the right choices.
- **Schema**:
    - `ModelProfile`: Local performance data.
    - `ModelBenchmark`: External leaderboard data.
    - `ModelFeedback`: User ratings.
    - `BenchmarkSync`: Tracking when we last updated data from HuggingFace.

---

## 5. Security & Production Readiness

While often used locally, we've added features to make the router safe for multi-user environments:

- **Rate Limiting**: Thread-safe request throttling protects your GPU from being overwhelmed by too many concurrent requests.
- **Admin Keys**: Protects sensitive endpoints like `/admin/reprofile` while keeping the main chat API accessible.
- **Sanitization**: All prompts are stripped of control characters and validated against length limits to prevent injection or memory-exhaustion attacks.
- **SQL Injection Prevention**: All database write operations use whitelist validation and ORM-based parameterized queries. Critical paths (e.g., `bulk_upsert_benchmarks`) explicitly validate keys against known column names.
- **Cascading Fallbacks**: If the "best" model happens to be down or fails mid-generation, the router can automatically retry with the "second best" model, improving overall system reliability.

---

## 5.1 Observability

SmarterRouter includes built-in observability features to help operators monitor and debug the system:

- **Structured Logging**: Set `ROUTER_LOG_FORMAT=json` to emit logs in JSON format. Each log entry includes a timestamp, level, logger name, message, and any extra context fields. A unique `request_id` is automatically added to all logs within the scope of a single HTTP request, enabling end-to-end tracing.

- **Request Correlation**: The `X-Request-ID` response header (and request header if provided) propagates a unique identifier across logs. This is invaluable for debugging complex multi-service interactions.

- **Prometheus Metrics**: The `/metrics` endpoint exposes a comprehensive set of counters and gauges:
  - `smarterrouter_requests_total` (labels: endpoint, method)
  - `smarterrouter_request_duration_seconds` (labels: endpoint)
  - `smarterrouter_errors_total` (labels: endpoint, error_type)
  - `smarterrouter_model_selections_total` (labels: selected_model, category)
  - `smarterrouter_cache_hits_total` / `smarterrouter_cache_misses_total` (labels: cache_type)
  - `smarterrouter_vram_total_gb`, `smarterrouter_vram_used_gb`, `smarterrouter_vram_utilization_pct`
  - `smarterrouter_gpu_total_gb`, `smarterrouter_gpu_used_gb`, `smarterrouter_gpu_free_gb` (labels: gpu_index) for multi‑GPU systems

- **VRAM Dashboard**: The `/admin/vram` endpoint provides a JSON snapshot of current GPU memory, loaded models with estimates, and recent history. Useful for ad‑hoc inspection or building custom dashboards.

- **Health Checks**: `/health` indicates the process is alive. Consider also implementing your own external checks that verify backend connectivity and VRAM thresholds.

These features make it easy to integrate SmarterRouter into production monitoring stacks (Grafana, Datadog, etc.) without requiring external agents.

---

## 5.2 Smart Caching

The router implements a multi-layered caching system to optimize performance:

### Routing Cache (Semantic)
The routing cache provides two layers of lookup:

1. **Exact Hash Matching** (Always active when cache enabled)
   - Uses SHA-256 of the prompt for instant cache hits
   - **No embedding model required** - works out of the box
   - 100% exact matches return cached `RoutingResult` immediately
   - This is the primary cache mechanism; identical prompts are served instantly

2. **Semantic Similarity** (Optional, requires `ROUTER_EMBED_MODEL`)
   - Computes embeddings for the prompt and compares against cached embeddings
   - Uses cosine similarity with threshold (default: 0.85)
   - Allows semantically similar prompts to reuse routing decisions
   - Example: "How do I reverse a linked list?" may hit cache for "Explain linked list reversal algorithm"

- **LRU Eviction**: Maintains up to 500 routing entries with 1-hour TTL.
- **Thread-Safe Operations**: All cache access is protected by an `asyncio.Lock`, ensuring correct behavior under concurrent load.
- **Tracks Recent Selections**: Keeps track of model selection frequency for diversity awareness and prevents model monopolization.

### Profile & Benchmark Cache (Database Query Optimization)
To minimize database round-trips during routing decisions:

- **In-Memory TTL Cache**: Profile and benchmark data is cached in memory with a 60-second TTL
- **Pre-warming on Startup**: `RouterEngine.warmup_caches()` is called automatically during server initialization to eliminate first-request latency
- **Smart Invalidation**: Cache is automatically invalidated after benchmark sync completes, ensuring fresh data
- **Targeted Queries**: Uses `get_benchmarks_for_models(model_names)` instead of fetching all benchmarks, reducing query scope

This optimization prevents redundant database queries on every routing decision when the cache is warm.

### Response Cache
- **Full Response Caching**: Caches actual LLM responses, not just routing decisions.
- **Model-Specific Keys**: Cache key is (model_name, prompt_hash).
- **Separate Storage**: 200-entry cache to balance memory usage and hit rate.
- **Signature Handling**: Signatures are added after retrieving cached responses to prevent duplication.
- **Atomic Updates**: Cache writes are synchronized to prevent race conditions.

### Cache Management
- **Detailed Stats**: Hit rates, similarity hit rates, miss reasons tracked.
- **Invalidation API**: `POST /admin/cache/invalidate` for manual cache clearing.
- **Per-Model Invalidation**: Can clear cache for specific models only.

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_CACHE_ENABLED` | true | Enable/disable caching |
| `ROUTER_CACHE_MAX_SIZE` | 500 | Max routing cache entries |
| `ROUTER_CACHE_TTL_SECONDS` | 3600 | Time-to-live for entries |
| `ROUTER_CACHE_SIMILARITY_THRESHOLD` | 0.85 | Similarity threshold (0-1) |
| `ROUTER_CACHE_RESPONSE_MAX_SIZE` | 200 | Max response cache entries |
| `ROUTER_EMBED_MODEL` | - | Embedding model for semantic matching |

---

## 6. API Reference

The router implements a fully OpenAI-compatible API, allowing it to serve as a drop-in replacement for most AI applications.

### 6.1 Core Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/v1/chat/completions` | POST | Main chat endpoint. Routes prompts to the best available model. Supports streaming. |
| `/v1/embeddings` | POST | Generates vector embeddings for text input. Useful for RAG and semantic search. |
| `/v1/models` | GET | Lists available models (returns the router as a single model entry). |
| `/v1/skills` | GET | Lists available tools/skills for agentic workflows. |
| `/v1/feedback` | POST | Submit user feedback to improve future routing decisions. |

### 6.2 Admin Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/admin/profiles` | GET | View performance profiles of all profiled models. |
| `/admin/benchmarks` | GET | View aggregated benchmark data from external sources. |
| `/admin/reprofile` | POST | Trigger manual reprofiling of models. |
| `/admin/cache/invalidate` | POST | Invalidate cache entries. Parameters: `model` (optional), `response_cache_only` (bool). |

### 6.3 Chat Completion Parameters

The router supports all standard OpenAI generation parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | List of message objects. **Required.** |
| `model` | string | Optional model override. |
| `temperature` | float | Sampling temperature (0.0 - 2.0). |
| `top_p` | float | Nucleus sampling threshold (0.0 - 1.0). |
| `n` | integer | Number of chat completion choices to generate. |
| `max_tokens` | integer | Maximum tokens to generate. |
| `presence_penalty` | float | Repetition penalty (-2.0 - 2.0). |
| `frequency_penalty` | float | Frequency penalty (-2.0 - 2.0). |
| `logit_bias` | object | Modify likelihood of specific tokens. |
| `user` | string | End-user identifier for tracking. |
| `seed` | integer | Seed for reproducible outputs. |
| `logprobs` | boolean | Include token log probabilities in response. |
| `top_logprobs` | integer | Number of most likely tokens to return. |
| `stream` | boolean | Enable server-sent events streaming. |
| `tools` | array | List of tools the model may call. |
| `tool_choice` | string/object | Force specific tool or auto. |
| `response_format` | object | Require JSON output. |

### 6.4 Embeddings Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | **Required.** Embedding model to use. |
| `input` | string/array | Text or list of texts to embed. |
| `user` | string | End-user identifier. |
| `encoding_format` | string | `float` (default) or `base64`. |

### 6.5 Response Structure

**Chat Completion Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "llama3:8b-instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

**Embeddings Response:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, -0.456, ...],
    "index": 0
  }],
  "model": "nomic-embed-text",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 0,
    "total_tokens": 8
  }
}
```
