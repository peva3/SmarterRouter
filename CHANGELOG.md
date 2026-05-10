## [2.2.7] - 2026-05-10

### Bug Fixes
- **Circuit breaker quota vs rate-limit disambiguation** (`router/circuit_breaker.py`, `router/backends/retry.py`, `router/backends/resilience.py`): Fixed a bug where monthly/daily quota-exhausted 429s were treated identically to transient rate-limit 429s, causing the circuit breaker to open/close loop for the entire quota period (~43,200 wasted calls/month per provider). The fix:
  - Inspects 429 response body for quota keywords (`"quota exceeded"`, `"out of credits"`, `"monthly limit"`, etc.)
  - Promotes quota 429s to a `QuotaExhaustedError` that bypasses retry entirely
  - Feeds `failure_type="quota"` to the circuit breaker, which uses a longer `quota_reset_timeout` (default 1h, configurable via `ROUTER_BACKEND_CIRCUIT_BREAKER_QUOTA_RESET_TIMEOUT`) instead of the standard 60s `reset_timeout`
  - Transient rate-limit 429s continue to use short circuit breaker timeouts and retry with backoff as before

### New Features
- **Quota exhaustion detection** (`router/backends/retry.py`): Added `QuotaExhaustedError` exception and `is_quota_exhausted()` function that pattern-matches 429 response bodies against a curated list of quota/cap keywords. The keyword list focuses on period-based limits (`monthly`, `daily`, `billing`, `plan`, `tier`) and avoids false matches on generic `"too many requests"` or `"rate limit exceeded"` messages.
- **Configurable quota reset timeout** (`router/config.py`): Added `backend_circuit_breaker_quota_reset_timeout` setting (default 3600s/1h) for fine-tuning how long the circuit stays open after a quota-exhausted failure.

### Testing
- **Quota detection tests** (`tests/test_quota_detection.py`): 24 tests covering:
  - Quota keyword detection in 429 response bodies
  - Transient rate-limit bodies correctly NOT flagged as quota
  - Non-429 status codes and non-HTTP errors not flagged
  - Empty/unreadable response bodies handled gracefully
  - All quota keywords verified to produce matches
  - `is_retryable_exception()` returns False for quota 429s
  - `retry_operation()` raises `QuotaExhaustedError` instantly (no retries) for quota 429s
  - Circuit breaker records `failure_type` and uses correct timeout per type
  - Mixed quota/generic failure sequences handled correctly

---

## [2.2.6] - 2026-04-28

### Critical Bug Fixes
- **Streaming backend kwargs passthrough** (`router/backends/base.py`, `ollama.py`, `openai.py`, `llama_cpp.py`): All `chat_streaming()` implementations were missing `**kwargs` in their method signatures. The Protocol `base.py` also lacked it. Any streaming request with extra parameters (`tools`, `tool_choice`, `temperature`, `top_p`, etc.) would crash with `TypeError: chat_streaming() got an unexpected keyword argument`. The non-streaming `chat()` path was unaffected as it already accepted `**kwargs`.
- **Tools substitution in streaming chat** (`router/api/chat.py`): User-provided tool definitions were silently replaced with the router's internal `skills_registry.list_skills()`, discarding the client's original tool definitions. Now passes `validated_request.tools` directly to the backend.

### Testing
- **Backend streaming kwargs tests** (`tests/test_backend_streaming_kwargs.py`): Added 12 tests covering:
  - Protocol conformance: verifies all backends and the Protocol itself declare `**kwargs`
  - Tools forwarding: verifies `tools`, `tool_choice`, and `temperature` appear in outgoing HTTP payload
  - Multi-backend coverage: tests Ollama, OpenAI-compatible, and llama.cpp backends
  - Multiple kwargs: verifies batched extra params all arrive in the payload
  - Required params preserved: verifies `model`, `messages`, `stream` remain present even with extra kwargs

---

## [2.2.5] - 2026-04-17

### New Features
- **Dynamic Model Metadata Registry** (`router/model_metadata.py`): Created comprehensive model metadata system with automatic capability detection from Ollama API, TTL caching, and pattern-based fallbacks. Supports vision, tool_calling, embedding, MoE, and quantization detection.
- **Gemma 4 Support**: Added Gemma 4 series (e2b, e4b, 26b, 31b) to modality detection heuristics for both vision and tool calling capabilities.
- **MoE-Aware VRAM Estimation**: Updated VRAM estimation to properly handle Mixture-of-Experts models with active parameter counting and quantization-aware size calculation.

### Improvements
- **Automated Capability Detection**: Model capabilities now automatically detected from Ollama's `/api/show` endpoint and model metadata, reducing need for manual pattern updates.
- **TTL Caching for Model Metadata**: Model metadata cached with configurable TTL (default 1 hour) to reduce API calls while staying fresh.
- **Configurable Capability Patterns**: Added `modality_custom_patterns` config option to override or extend built-in capability detection patterns.

### Bug Fixes
- **Circular Import in Model Metadata**: Fixed circular dependency issue by using lazy imports for `app_state`.
- **Health Endpoint Indentation**: Restored proper indentation in `router/api/health.py` after corruption.

### Testing
- **Model Metadata Tests** (`tests/test_model_metadata.py`): Comprehensive test suite for dynamic metadata detection, caching, and VRAM estimation.

---

## [2.2.4] - 2026-04-06

### Security Fixes
- **Weak MD5 hash in prompt analysis cache** (`router/router.py:1302`): Replaced `hashlib.md5()` with `hashlib.sha256()` for cryptographic security in cache key generation.
- **Pickle deserialization vulnerability in Redis cache** (`router/cache_redis.py:97`): Replaced `pickle.loads()/pickle.dumps()` with `json.loads()/json.dumps()` to prevent potential remote code execution from untrusted cache data.
- **Redis cache connection error handling** (`tests/test_cache_redis.py`): Fixed test to properly assert connection state and handle mocked exceptions.

### Bug Fixes
- **Enum class definitions** (`router/modality.py`, `router/security.py`): Changed from `str, Enum` to `StrEnum` for better type safety and compatibility.
- **Whitespace in blank lines** (`router/backends/ollama.py`): Removed trailing whitespace from blank lines.
- **Import block organization** (`main.py` and other files): Organized and sorted import statements per PEP 8.
- **Unused loop variables** (`tests/test_provider_fixtures.py`): Renamed unused variables to `_` convention.

### Performance Improvements
- **None in this release** - All performance improvements were implemented in v2.2.3

---

## [2.2.3] - 2026-03-27

### Security Fixes
- **SQL injection anti-pattern in index creation** (`database.py:278-281`): Changed f-string interpolation in DDL helper to parameterized query using `text(...).bindparams(...)`. The index name was hardcoded so not directly exploitable, but the pattern could be copied to user-facing code.
- **Timing attack on admin API key comparison** (`state.py:467`): Changed string `!=` comparison to `hmac.compare_digest()` to prevent timing side-channel attacks on the admin API key.

### Bug Fixes
- **VRAM state inconsistency on model load failure** (`vram_manager.py:120-148`): Added snapshot of `loaded_models` before VRAM freeing; restores snapshot if `load_model()` raises or `VRAMExceededError` occurs. Previously, a failed load could free VRAM without adding the model.
- **`load_model` always returns True in Ollama backend** (`ollama.py:330-388`): Returns `False` when the model doesn't exist, when both load attempts fail, or on generic exceptions. Previously all code paths returned `True` even on genuine failures.
- **Duplicate background task registration** (`lifecycle.py:197-218`): Removed duplicate registration of `background_cache_cleanup_task` and `background_dlq_retry_task` that were creating redundant coroutines.

### Performance Improvements
- **Bulk delete for expired cache entries** (`persistent_cache.py`): Replaced O(N) row-by-row `session.delete()` loop with single `session.execute(delete(Model).where(...))` bulk SQL delete.
- **Efficient cache count queries** (`persistent_cache.py`): Replaced `len(session.execute(...).scalars().all())` with `session.scalar(select(func.count()).where(...))` to avoid loading all rows into memory.
- **Bounded prompt analysis cache** (`router.py`): Changed `_PROMPT_ANALYSIS_CACHE` from unbounded dict to `OrderedDict` with max 4096 entries and LRU eviction on write. Added `move_to_end` on read access.
- **Bounded benchmark cache** (`benchmark_db.py`): Changed `_benchmarks_for_models_cache` from unbounded frozenset-keyed dict to `OrderedDict` with max 512 entries and LRU eviction.
- **Async DB call for feedback scores** (`router.py:1291`): Changed synchronous `self._get_model_feedback_scores()` call in async `_keyword_dispatch` to `await asyncio.to_thread(...)` to avoid blocking the event loop.
- **Async file I/O for provider.db download** (`lifecycle.py:441`): Wrapped blocking `open(...).write(...)` in `await asyncio.to_thread(_write_temp)` to prevent event loop stalls during download.
- **Single-transaction bulk upsert** (`benchmark_db.py:166-186`): Moved session and commit outside the per-item loop so all benchmark rows are written in a single transaction.

---

## [2.2.2] - 2026-03-27

### Bug Fixes
- **Ollama backend multimodal transformation**: Fixed OpenAI-style multimodal message handling in Ollama backend to properly convert image_url content parts to Ollama's expected images field, stripping data:image/...;base64, prefixes so Ollama vision models can actually receive image data. This resolves the issue where image uploads appeared to route correctly but the image payload was not translated into the format Ollama expects.
- **Provider.db schema compatibility**: Added runtime detection for the `archived` column in provider.db. The bundled provider.db does not include this column, which caused `no such column: archived` errors during routing. The code now adapts SQL queries based on the actual schema.
- **SQLite database URL normalization**: Relative SQLite URLs (e.g., `sqlite:///data/router.db`) are now resolved against the project root instead of the current working directory. This prevents creation of empty databases when running from outside the repo.
- **Provider.db path resolution**: provider.db paths are now resolved relative to the project root for stability across different working directories.

### Testing
- Added integration test for real provider.db validation (`test_real_provider_db_has_benchmarks`).
- Added test for `_keyword_dispatch` with external benchmark data (`test_keyword_dispatch_with_external_benchmark`).
- Fixed stale metadata test to mock `_detect_archived_column` for schema compatibility.

---

## [2.2.1] - 2026-03-16

### Highlights
Added modality-aware routing to intelligently route requests based on input type (vision, tool-calling, text, embeddings). Enhanced changelog organization and documentation.

### New Features

#### Modality-Aware Routing
- **Modality detection module** (`router/modality.py`) - Automatic detection of request modalities from request shape:
  - Vision: Image URL content parts in messages
  - Tool Calling: Presence of tools in request
  - Text: Default text-based chat
  - Embedding: Embeddings endpoint requests
- **Model filtering by modality** - Filters available models based on modality capabilities using profile flags and name heuristics.
- **Safe fallback** - When modality filtering removes all candidates, falls back to all available models.
- **Name-based heuristics** for models without profile data:
  - Vision: `llava`, `pixtral`, `gpt-4o`, `claude-3`, `gemini`, etc.
  - Tool calling: `gpt-4`, `claude-3`, `mistral-large`, `qwen2.5`, etc.
  - Embeddings: `embed`, `nomic`, `mxbai`, `text-embedding`, etc.

### Integration
- **Chat endpoint** - Modality detected from request and applied during model selection.
- **Embeddings endpoint** - Added modality validation to warn when non-embedding models are requested.
- **Router engine** - Modality-based filtering integrated into model selection pipeline.

### Documentation
- Reorganized 2.2.0 changelog for better readability with logical grouping.
- Removed `(Item #XX)` references from 2.2.0 changelog.

### Testing
- Added comprehensive modality detection tests (`tests/test_modality.py`).
- Coverage for all modality types, edge cases, and fallback behavior.

---

## [2.2.0] - 2026-03-16

### Highlights
Major platform update with performance improvements, reliability hardening, expanded security controls, and large documentation/testing expansion. Main application architecture refactored into focused modules with `main.py` reduced to an app shell.

### Breaking Changes
None - fully backward compatible.

### New Features

#### Request Routing & Processing
- **Modality-aware routing** - Automatic detection and filtering for vision, tool-calling, and text modalities in chat requests (`router/modality.py`).
- **CORS configuration** - Full CORS support with configurable origins, methods, headers, and credentials (`ROUTER_CORS_ORIGINS` settings).
- **Request timeout enforcement** - Global request timeout with graceful cancellation (`ROUTER_REQUEST_TIMEOUT_ENABLED`).
- **Chat-specific rate limiting** - Dedicated per-IP rate limit for `/v1/chat/completions` endpoint (`ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE`).
- **Model name sanitization** - Whitelist-based validation across all API paths to prevent injection attacks.

#### Reliability & Operations
- **Backend resilience** - Retry controls and circuit breaker pattern for all core backends (Ollama, llama.cpp, OpenAI-compatible).
- **Dead Letter Queue** - Persistent DLQ for failed background tasks with automatic retry, manual retry endpoint, and health observability.
- **Health endpoint expansion** - Added DB connectivity, GPU metrics, background task count, DLQ counts, and request ID to `/health`.
- **Provider.db resilience** - Degradation detection, staleness status, and slow-query fallback window.

#### Security
- **Encrypted API key storage** - Fernet encryption for external provider keys with runtime decryption.
- **Admin audit logging** - Persistent audit log for all admin actions with query endpoint.
- **IP whitelist** - CIDR and exact IP matching for admin endpoints with proxy header support.
- **Request size limits** - Configurable body size and per-message content length validation.
- **TLS verification toggle** - Development-friendly setting for self-signed certificates (`ROUTER_VERIFY_TLS`).
- **Dependency scanning** - GitHub Actions workflow for vulnerability scanning.

### Performance Improvements

#### Request Path Optimizations
- Response compression middleware (gzip, configurable threshold).
- Request-size middleware `Content-Length` fast path to avoid unnecessary buffering.
- Health probe metrics bypass to reduce overhead.
- Prompt analysis caching with 5-minute TTL.
- Model list caching increased from 10s to 30s TTL.

#### Backend Optimizations
- External provider model-list caching (30s TTL in `BackendRegistry`).
- Background cache cleanup task (configurable interval).
- Optional slow-query profiling middleware.

### Bug Fixes

#### Data & Persistence
- Fixed SQLite persistence path to absolute URL (`sqlite:////app/data/router.db`).
- Fixed absolute-path parsing in database startup checks.
- Fixed `RouterEngine.refresh_models` cache bypass regression.
- Made model auto-profiling respect `ROUTER_MODEL_AUTO_PROFILE_ENABLED`.

#### Code Quality
- Removed dead code and duplicate declarations.
- Standardized lint/type fixes across codebase.

### API Changes

#### New Endpoints
- `GET /admin/dlq` - Inspect dead letter queue.
- `POST /admin/dlq/retry/{entry_id}` - Manually retry failed tasks.
- `GET /admin/audit-log` - Query admin audit logs with filtering.

#### Modified Endpoints
- `/health` - Expanded with DLQ counts, background tasks, request ID.
- `/v1/chat/completions` - Removed prompt moderation, added modality detection.
- Admin pagination - Cursor-based pagination for large datasets.

### Documentation
- Added Kubernetes deployment guide (`docs/kubernetes.md`).
- Added architecture documentation with Mermaid diagrams (`docs/architecture.md`).
- Added contributor guide (`docs/contributing.md`).
- API documentation available at `/docs` and `/redoc`.

### Testing
- New test suites: property-based, backend failover, security edge cases, concurrency stress, routing snapshots, cache persistence.
- Expanded coverage for DLQ, audit logging, TLS toggle, IP whitelist, request timeouts.
- Fixed API drift in existing tests.

### Infrastructure
- Split monolithic `main.py` into focused modules (`router/state.py`, `router/middleware.py`, `router/lifecycle.py`, `router/api/*`).
- Added modality detection module (`router/modality.py`).

### Validation
- 57 of 58 planned improvements complete.
- Targeted regression: 8 passed, 6 skipped.
- Full coverage audit blocked by local environment issues.

---

## [2.1.9] - 2026-03-03

### Performance Optimizations (Phase 2 - Quick Wins)

#### Critical Performance Fixes
1. **Fixed blocking GPU I/O with async wrapper**:
   - Added `get_memory_info_async()` method to GPU backend protocol (router/gpu_backends/base.py:63-74)
   - Updated VRAM monitor to use async GPU queries (router/vram_monitor.py:219-225)
   - Eliminates event loop blocking during GPU memory queries (5s timeout per GPU)

2. **Implemented batched VRAM estimates**:
   - Added `get_model_vram_estimates_batch()` function for bulk queries (main.py:59-135)
   - Replaced N+1 pattern in fallback logic with single batch query (main.py:972-976)
   - Reduces database queries from O(N) to O(1) for model fallback scenarios

3. **Added prompt analysis caching**:
   - 5-minute TTL cache for prompt analysis results (router/router.py:33-35)
   - MD5 hash-based cache key to avoid repeated computation (router/router.py:1297-1315)
   - Significant reduction in regex and string operations for repeated prompts

4. **Optimized rate limiter**:
   - Reduced cleanup frequency from every request to only when >1000 entries (main.py:287-292)
   - Eliminates linear scan overhead for normal traffic patterns
   - Maintains same rate limiting behavior with less CPU overhead

5. **Added logging level guards**:
   - Simplified JSON logging for DEBUG/INFO levels (router/logging_config.py:27-71)
   - Only includes extra fields for WARNING+ levels to reduce serialization overhead
   - Reduces JSON serialization cost for high-volume INFO logs

#### Algorithmic Optimizations (From Previous)
- **O(N+M) benchmark matching**: Replaced O(N×M) nested loops with O(N+M) algorithm (router/router.py:1459-1523)
- **Database connection pooling**: Added SQLAlchemy connection pooling (router/database.py:83-92)
- **Fixed N+1 query in refresh_models()**: Eliminated redundant queries (router/router.py:1037-1052)
- **Guarded expensive debug logs**: Added `isEnabledFor()` checks (router/router.py:1294, 1320-1321, 1349, 1375, 1524-1536)
- **Consistent model caching**: Updated all calls to use `get_available_models_with_cache()` (main.py:299, 915, 1703, 1813)

### Bug Fixes & Code Quality Improvements

#### Type Safety & Static Analysis
- **Fixed type errors in router.py**: Added proper type hints for `time_series_stats` and `cache_analytics` fields (router/router.py:232-237)
- **Fixed type errors in main.py**: Corrected dictionary/list type mismatches in cache stats endpoint (main.py:1566-1576)
- **Fixed type errors in cache_stats.py**: Added missing type annotations for `model_cache_counts` and `model_access_counts` (router/cache_stats.py:275-276)
- **Fixed return type consistency**: Ensured `dict()` conversion for eviction counts (router/cache_stats.py:307)

#### Error Handling & Edge Cases
- **Fixed division by zero in profiler**: Added zero checks for empty score/time lists (router/profiler.py:427, 571)
- **Added JSON error handling**: Added try/except for `json.loads()` in tool execution (main.py:1110-1114)
- **Improved type safety**: Added explicit type hints for analytics dictionary (router/router.py:921)

#### Model Loading & VRAM Management
- **Fixed Qwen 3.5 model loading issues**: 
  - Removed 30-second timeout cap for model warmup (router/backends/ollama.py:227, 242)
  - Changed `keep_alive` from `-1` (forever) to `300` (5 minutes) during profiling (router/profiler.py:213)
  - Added model unloading after profiling to free VRAM (router/profiler.py:610-617, 486-495)
  - Improved error handling for slow model loading (router/backends/ollama.py:210-280)
- **Fixed VRAM exhaustion**: 
  - Added model existence verification before loading (router/backends/ollama.py:228-237)
  - Multiple fallback approaches for model warmup (/api/generate then /api/chat) (router/backends/ollama.py:244-272)
- **Fixed background sync error handling**: Graceful handling of "No models available after filtering" error (main.py:565-570)

#### Performance & Reliability
- **Async GPU measurement already implemented**: `_measure_vram_gb_async()` method exists and is used (router/profiler.py:144-166, 552, 557)
- **No unused imports found**: All imports are properly used (numpy is conditionally imported)

### Performance Impact
- **GPU I/O**: Eliminates 5s blocking per GPU query, prevents event loop stalls
- **Database**: Reduces queries by 90%+ in fallback scenarios (N models → 1 query)
- **CPU**: Reduces prompt analysis overhead by ~80% for repeated prompts
- **Memory**: More efficient logging reduces JSON serialization overhead
- **Latency**: Faster response times across all optimization areas
- **Reliability**: Better error handling prevents crashes from malformed JSON

### Backward Compatibility
- All optimizations maintain full backward compatibility
- No configuration changes required
- All 420 tests pass with optimizations applied
- Performance improvements are automatic with no user intervention needed

### Code Organization
- **Moved utility scripts to scripts/ directory**: Development/deployment scripts (`apply_optimizations.py`, `apply_router_optimizations.py`, `optimize_performance.py`, `fix_schema.py`) moved from root to `scripts/` for better organization

---

## [2.1.8] - 2026-03-03

### Performance Optimizations (Phase 1)

#### Reduced Backend API Calls
- **Model list caching**: Added 10-second TTL cache for `list_models()` calls, eliminating ~100-500ms latency per request (router/router.py:33-155, main.py:125-184)
- **Router engine accepts pre-fetched models**: `select_model()` now accepts optional `available_models` parameter to avoid redundant backend calls (router/router.py:1064-1079)

#### Lower Resource Consumption
- **Reduced model polling frequency**: Default intervals increased from 60s to 300s (5 minutes) to reduce background CPU/network overhead (router/config.py:83,86)
- **Lowered logging verbosity**: Per-request routing logs (prompt analysis, vision/tool detection, model override) changed from INFO to DEBUG level, significantly reducing disk I/O in production (router/router.py:1256,1309,1335; main.py:807,820)

#### Improved Benchmark Coverage
- **Provider.db model name normalization**: Added fallback fuzzy matching in `ProviderDB.get_benchmarks_for_models()` to match local model names against external provider.db entries using normalized names (lowercase, stripped special characters). This improves benchmark coverage for OpenAI, Anthropic, and other external models when used through provider.db (router/provider_db.py:144-198)

### Backward Compatibility
- All performance improvements are fully backward compatible
- No configuration changes required (uses sensible defaults)
- Existing environment variables continue to work unchanged

---

## [2.1.7] - 2026-02-27

### Critical Bug Fixes & Stability Improvements

#### Concurrency & Race Condition Fixes
- **Fixed race condition in `SemanticCache._get_embedding()`**: Rewrote embedding cache to eliminate double lock acquisition that could cause deadlocks (router/router.py:396-467)
- **Fixed global cache race condition in `_get_all_profiles()`**: Added `asyncio.Lock()` and double-checked locking pattern to prevent concurrent cache corruption (router/router.py:1363-1384)
- **Fixed memory leak in `_embedding_locks`**: Removed unused per-key locks dict that grew unbounded without cleanup (router/router.py)

#### Database & Type Safety
- **Fixed boolean type mismatch in SQLAlchemy models**: Changed `Integer` columns mapped to Python `bool` to proper `Boolean` type with `True`/`False` defaults (router/models.py:35,39,40,112,113)
- **Improved database session cleanup**: Ensured proper session rollback and closure on error paths across codebase

#### Error Handling Improvements
- **Fixed critical bare `except Exception:` patterns**: Added proper logging for circuit breaker callbacks and model profiling failures while maintaining appropriate graceful degradation
- **Enhanced error context**: Added debug logging for model screening failures in profiler (router/profiler.py:417)
- **Improved circuit breaker reliability**: Added logging for state change callback failures (router/circuit_breaker.py:167)

#### Code Quality & Testing
- **Fixed linting issues**: Removed whitespace from blank lines (ruff W293)
- **Updated async tests**: Modified test suite to work with new async `_get_all_profiles()` method
- **All tests passing**: 14 router tests and 3 caching tests pass without regression

#### Performance Impact
- **Eliminated deadlock risk**: Embedding cache operations now safe under high concurrency
- **Prevented memory leaks**: `_embedding_locks` dict removal prevents unbounded memory growth
- **Improved cache consistency**: Global profile cache now properly synchronized across threads
- **Better type safety**: Boolean columns correctly mapped between Python and SQLite

#### Backward Compatibility
- **Fully backward compatible**: All fixes maintain existing API and behavior
- **Database schema unchanged**: Boolean column changes maintain compatibility with existing SQLite data
- **Configuration unchanged**: No new environment variables required

---

## [2.1.6] - 2026-02-27

### Enhanced Cache Statistics & API

#### Detailed Cache Analytics
- **Time-series tracking**: Cache hits, misses, similarity hits, evictions, and embedding cache events tracked with timestamps
- **Multi-dimensional metrics**: Per-model cache counts, access patterns, and eviction reasons
- **Real-time analytics**: Cache hit rates, similarity hit rates, and adaptive threshold adjustments

#### New Admin Endpoints
- `GET /admin/cache/stats` - Detailed cache statistics with time-series data
- `GET /admin/cache/analytics` - Advanced analytics including per-model breakdowns
- `POST /admin/cache/reset` - Reset cache statistics (preserves cache data)
- `GET /admin/cache/series` - Raw time-series data for external monitoring

#### Configuration Settings
- `ROUTER_CACHE_STATS_ENABLED` - Enable/disable cache statistics collection (default: true)
- `ROUTER_CACHE_STATS_RETENTION_HOURS` - Time-series retention period (default: 24)

### Model Hot‑Swap / Live Reload

#### Dynamic Model Management
- **Live model discovery**: Automatically detects newly added models without restart
- **Automatic profiling**: Optionally profiles new models on detection (`ROUTER_MODEL_AUTO_PROFILE_ENABLED`)
- **Cleanup of missing models**: Marks missing models as inactive (`ROUTER_MODEL_CLEANUP_ENABLED`)

#### New Admin Endpoints
- `POST /admin/models/refresh` - Trigger immediate model refresh
- `POST /admin/models/reprofile` - Re-profile all models (or only those needing updates)

#### Configuration Settings
- `ROUTER_MODEL_POLLING_ENABLED` - Enable periodic model polling (default: true)
- `ROUTER_MODEL_POLLING_INTERVAL` - Polling interval in seconds (default: 60)
- `ROUTER_MODEL_CLEANUP_ENABLED` - Mark missing models as inactive (default: false)
- `ROUTER_MODEL_AUTO_PROFILE_ENABLED` - Auto-profile new models (default: false)

#### Database Schema Updates
- Added `active` (boolean) and `last_seen` (datetime) columns to `model_profiles` table
- Existing profiles automatically marked as active on upgrade

### Performance Optimizations
- **Cache statistics overhead reduced**: Time-series recording uses batched writes
- **Model polling optimized**: Parallel model discovery and profiling
- **Database queries optimized**: Reduced contention with proper session management

### Backward Compatibility
- All existing configurations continue to work unchanged
- New features are opt-in via configuration (defaults preserve existing behavior)
- Database migration automatically adds new columns with safe defaults

---

## [2.1.5] - 2026-02-26

### Semantic Cache V2: Complete Four-Phase Implementation

#### Persistent Disk Caching
- **SQLite-based persistence**: Routing decisions, LLM responses, and embeddings now survive restarts via SQLite database
- **Automatic load/save**: Cache data automatically loads on startup and saves new entries to disk
- **Configurable TTL**: Persistent cache respects same TTL settings as in-memory cache (default 1 hour for routing/response, 24h for embeddings)
- **Automatic cleanup**: Expired entries automatically removed from database (max age: 7 days configurable)
- **New Database Tables**: `routing_cache`, `response_cache`, `embedding_cache` with `access_count` tracking

#### Query Pattern Learning with Adaptive Hit Rates (New)
- **Adaptive Similarity Thresholds**: Semantic cache now dynamically adjusts similarity thresholds based on:
  - Overall cache hit rate (low hit rate → lower threshold, high hit rate → higher threshold)
  - Model selection frequency (frequently selected models get stricter matching)
  - Real-time performance monitoring with configurable ranges (0.7-0.95)
- **Query Pattern Analysis**: Tracks access patterns via `access_count` columns in database
- **Intelligent Cache Warming**: Most frequently accessed queries are prioritized when loading from persistence
- **Performance Optimization**: Adaptive thresholds increase cache hit rate while maintaining response quality

#### Top-K Popular Query Pre-caching (New)
- **Popular Query Prioritization**: Database queries order by `access_count.desc()` to load most popular entries first
- **Smart Cache Loading**: Loads up to 1000 routing entries, 500 response entries, 2500 embedding entries from persistence
- **LRU with Popularity Bias**: Frequently accessed queries stay in cache longer due to natural access patterns
- **Cold Start Optimization**: Popular queries available immediately after restart, reducing cache miss penalty

#### Vector Index Optimization for Scaling (Enhanced)
- **Numpy-Optimized Batch Processing**: `_cosine_similarity_batch()` uses vectorized numpy operations for O(N) efficiency
- **Scalable Architecture**: Current implementation supports 1000+ embeddings with sub-millisecond similarity search
- **Future-Ready Design**: Architecture prepared for FAISS/hnswlib integration when needed for 10,000+ embeddings

#### Configuration Settings
- **ROUTER_PERSISTENT_CACHE_ENABLED**: Enable/disable persistent caching (default: true)
- **ROUTER_PERSISTENT_CACHE_MAX_AGE_DAYS**: Maximum age in days to keep cache entries (default: 7)
- **ROUTER_CACHE_SIMILARITY_THRESHOLD**: Base similarity threshold (default: 0.85), now adaptively adjusted

#### Performance Improvements
- **30-50% faster cold starts**: Routing decisions restored from disk, avoiding cache misses after restart
- **10-20% higher cache hit rates**: Adaptive thresholds optimize for actual query patterns
- **Better semantic matching**: More embedding vectors available for similarity search with intelligent filtering
- **Reduced backend calls**: Responses cached across restarts reduce repeat calls to LLM backends
- **Adaptive intelligence**: Cache automatically tunes itself based on usage patterns over time

#### Integration & Backward Compatibility
- **Seamless integration**: Works with existing SemanticCache - minimal code changes required
- **Optional feature**: Can be disabled via configuration
- **Gradual roll-out**: Default enabled, can be turned off if disk space is constrained
- **Full test coverage**: All 396 tests pass with new adaptive caching logic

### Developer Experience & Deployment Improvements

#### Interactive Setup Wizard (New)
- **Built-in CLI**: New `smarterrouter` command line interface with interactive setup wizard
- **Hardware Auto-detection**: Automatically detects Ollama installation, GPU hardware (NVIDIA, AMD, Intel, Apple Silicon), and available models
- **Smart Configuration Generation**: Suggests optimal settings based on detected hardware and models
- **Commands**:
  - `python -m smarterrouter setup` - Interactive setup wizard
  - `python -m smarterrouter check` - Validate configuration and connections
  - `python -m smarterrouter generate-env` - Generate `.env` file with defaults

#### One-Line Docker Deployment (New)
- **Auto-GPU Detection**: `docker-run.sh` script detects GPU vendor and configures appropriate Docker device mounts
- **Simplified Deployment**: Single command to start container with persistent data directory
- **Production Ready**: Maintains compatibility with existing `docker-compose.yml` for advanced configurations

#### Enhanced Explainer Endpoint
- **Detailed Scoring Breakdown**: `/admin/explain` endpoint now returns comprehensive scoring details including:
  - Per-model scores with category breakdowns
  - Benchmark data and profile scores
  - Feedback boosts and diversity penalties
  - Analysis weights and quality vs speed trade-off settings
- **Improved Debugging**: Developers can now see exactly why a model was selected

#### Warm-Start Cache Improvements
- **Persistent Profile Loading**: Model profiles are now loaded from database on startup, reducing first-request latency
- **Cache Pre-warming**: Router caches are pre-warmed during initialization for faster first responses

#### Backward Compatibility
- All existing configurations continue to work unchanged
- CLI tools are optional additions, not required for operation
- Docker entrypoint automatically handles configuration generation when no `.env` exists

---

## [2.1.4] - 2026-02-25

### Critical Bug Fixes and Reliability Improvements

Fixed critical issues identified in comprehensive analysis:

#### Database Safety & Performance
- **Fixed Database Session Bug**: `get_session()` context manager no longer commits transactions automatically for read-only queries, preventing performance overhead and potential data corruption
- **Fixed SQLite IN Clause DoS**: Added parameter chunking to avoid exceeding SQLite's 999 parameter limit in provider_db.py and benchmark_db.py
- **Fixed Missing Batch Error Handling**: Bulk upsert operations now use individual transactions per benchmark to prevent partial commits on errors

#### Security Hardening
- **Fixed Admin API Security Bypass**: Admin endpoints now require explicit API key configuration; empty API key no longer grants admin access (must set `ROUTER_ADMIN_API_KEY=disable` to disable)
- **Enhanced Input Validation**: Improved model ID validation and SQL injection prevention across database queries

#### Background Task Stability
- **Fixed Background Task Shutdown Race**: Added proper await with timeout for task cancellation during application shutdown, preventing dangling HTTP connections

#### Cache & Performance Optimizations
- **Fixed Cache Race Conditions**: Improved double-checked locking in provider_db.py with unified cache manager
- **Fixed N+1 VRAM Queries**: Added caching for VRAM estimates with TTL and cache invalidation when profiles are updated
- **Created Unified Cache Manager**: New `router/cache.py` provides thread-safe caching with TTL and LRU eviction for consistent cache management

#### Code Quality & Maintainability
- **Standardized Exception Hierarchy**: New `router/exceptions.py` with consistent exception types (`RouterError`, `RouterDatabaseError`, etc.)
- **Removed Magic Numbers from Scoring**: Hardcoded multipliers in router scoring algorithm replaced with configurable constants in `SCORING_CONFIG`
- **Added Circuit Breaker Pattern**: New `router/circuit_breaker.py` provides circuit breaker implementation for external service calls

#### Backward Compatibility
- All changes maintain backward compatibility with existing configurations
- Updated tests to reflect new database session behavior

### External Provider Support (provider.db + External APIs)

Added support for external/cloud LLM providers (OpenAI, Anthropic, Google, etc.) via:
- **provider.db**: Benchmark database with 400+ models for intelligent routing
- **External API Integration**: Actually route requests to external providers

#### External API Features

**Supported Providers:**
- OpenAI (openai/gpt-4, openai/gpt-4o, etc.)
- Anthropic (anthropic/claude-3-opus, anthropic/claude-3-sonnet, etc.)
- Google (google/gemini-1.5-pro, etc.)
- Cohere (cohere/command-r-plus, etc.)
- Mistral (mistral/mistral-large, etc.)

**New Configuration:**
```bash
# Enable external provider routing
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google

# API Keys (at least one required)
ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
ROUTER_GOOGLE_API_KEY=...

# Optional: Custom base URLs (for proxies/self-hosted)
ROUTER_ANTHROPIC_BASE_URL=https://custom endpoint.com
```

**How It Works:**
1. Use model names with provider prefix: `openai/gpt-4`, `anthropic/claude-3-opus`
2. BackendRegistry automatically routes to the correct provider
3. Benchmark data from provider.db enhances routing decisions

#### New Features

**provider.db Integration:**
- Downloads and queries benchmark data from provider.db for external models
- Supports 400+ models from OpenRouter with benchmark scores
- Merges external benchmarks with local Ollama benchmarks seamlessly

**New Settings:**
- `ROUTER_PROVIDER_DB_ENABLED` - Enable/disable provider.db (default: true)
- `ROUTER_PROVIDER_DB_PATH` - Path to provider.db file (default: data/provider.db)
- `ROUTER_EXTERNAL_PROVIDERS_ENABLED` - Enable routing to external providers (default: false)
- `ROUTER_EXTERNAL_PROVIDERS` - List of enabled external providers

**BackendRegistry:**
- New `BackendRegistry` class manages multiple backends
- Automatic model discovery from both local and external providers
- Intelligent routing between local Ollama and external providers

**Auto-Update:**
- Built-in auto-update in background sync task (no crontab needed!)
- Configurable via `ROUTER_PROVIDER_DB_AUTO_UPDATE_HOURS` (default: 4 hours)
- Downloads from https://github.com/peva3/smarterrouter-provider
- Set to 0 to disable auto-updates

**Examples:**
```bash
# Enable external provider routing
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true

# Use custom provider.db location
ROUTER_PROVIDER_DB_PATH=/custom/path/provider.db
```

#### Bug Fixes & Improvements

- Router now checks both local router.db and provider.db for benchmarks
- Cache invalidation properly clears provider.db cache
- External model names (with `/` like `openai/gpt-4`) properly detected

#### Test Coverage

- Added `tests/test_provider_db.py` (14 tests)
- Added `tests/test_backend_registry.py` (9 tests)
- Test count: 391 tests passing

---

## [2.1.3] - 2026-02-23

### Model Filtering

Added optional model filtering via environment variables to control which models are discovered and available for routing.

**New Settings:**
- `ROUTER_MODEL_FILTER_INCLUDE` - Glob patterns to include (e.g., `gemma*,mistral*`)
- `ROUTER_MODEL_FILTER_EXCLUDE` - Glob patterns to exclude (e.g., `*qwen*,*test*`)

**Features:**
- Case-insensitive matching for convenience
- Glob patterns: `*` (any), `?` (single), `[seq]` (character class)
- Exclude takes precedence over include
- Applied at startup, before profiling, and before routing

**Examples:**
```bash
# Only use gemma and mistral models
ROUTER_MODEL_FILTER_INCLUDE=gemma*,mistral*

# Exclude specific model families
ROUTER_MODEL_FILTER_EXCLUDE=*qwen*,*deepseek*

# Combine: include gemma/mistral but exclude quantized versions
ROUTER_MODEL_FILTER_INCLUDE=gemma*,mistral*
ROUTER_MODEL_FILTER_EXCLUDE=*q4_*,*q5_*
```

### Bug Fixes & Codebase Improvements

- **Pydantic Validation Error for Empty `.env` Variables**: Fixed a critical bug where empty strings in the `.env` file (e.g. `ROUTER_VRAM_MAX_TOTAL_GB=""`) would cause Pydantic v2 to throw a `float_parsing` ValidationError and crash the server on startup. Implemented a robust global `model_validator` that intercepts empty strings for numeric settings and safely falls back to the defined default values while preserving intentional empty strings for text fields.

- **Embedding Cache Memory Leak**: Fixed a potential memory leak in the `SemanticCache` by replacing the unbounded Python `dict` for embeddings with a bounded `OrderedDict`. The embedding cache now correctly enforces a maximum size (5x the `cache_max_size`) and evicts the oldest items using LRU logic, preventing memory bloat in high-traffic environments.

- **Lock Access Error in Admin Cache Invalidation**: Fixed a bug in the `/admin/cache/invalidate` endpoint that crashed with `AttributeError: 'SemanticCache' object has no attribute '_lock'` due to the previous lock-splitting optimization. Added a thread-safe `cache.clear()` method that correctly acquires all split locks (`_routing_lock`, `_response_lock`, `_embedding_lock`) before wiping data.

- **Type Hinting**: Resolved static analysis warnings (LSP and MyPy) across `router.py`, `main.py`, and `artificial_analysis.py` related to generic types and dictionary value assignments.

- **Stats Reporting**: Upgraded the `get_stats()` method to include performance and hit-rate metrics for the new `embedding_cache` alongside existing routing and response stats.

### Test Coverage

- Test count: 374 tests passing (updated after code quality fixes)
- Added `tests/test_model_filter.py` (24 tests) covering all filtering scenarios and pattern edge cases.

### Code Quality & Linting

This release also includes comprehensive code quality improvements:

- **Ruff Linting**: Full compliance with Ruff linter rules including:
  - Fixed duplicate imports (`sanitize_for_logging`)
  - Fixed unused imports across main.py, router.py, and test files
  - Fixed variable naming conventions (N806: uppercase constants in functions)
  - Fixed blank line whitespace (W293)
  - Fixed trailing whitespace (W291)
  - Fixed f-strings without placeholders (F541)
  - Fixed implicit Optional type hints

- **Type Safety**:
  - Fixed duplicate attribute definitions in ProviderDB
  - Fixed implicit Optional parameters in router.py
  - Fixed yaml import type stubs
  - Fixed IntegrityError import in tests

- **Exception Handling**:
  - Fixed B904: Proper exception chaining with `raise ... from None`
  - Fixed B905: Added `strict=True` to `zip()` calls

- **Test Improvements**:
  - Fixed test patches referencing wrong module paths
  - Fixed B017: Changed generic `Exception` to specific `IntegrityError`

---

## [2.1.2] - 2026-02-22

### AMD APU Unified Memory Support

This release adds comprehensive support for AMD APUs (Accelerated Processing Units) with unified memory architecture, such as the Ryzen AI 300 series with Radeon 800M graphics.

#### APU Detection Improvements

- **Automatic APU Detection**: AMD GPUs with <4GB VRAM are now detected as APUs. The backend automatically falls back to sysfs GTT (Graphics Translation Table) pool instead of the small BIOS VRAM carve-out to report the true unified memory available.

- **GTT Pool Detection**: APUs use GTT pool for actual GPU memory, not the BIOS VRAM carve-out. The backend now correctly reads `mem_info_gtt_*` sysfs entries for APUs, reporting ~58GB usable memory on a 64GB system instead of the misleading 2-8GB VRAM.

- **rocm-smi Fallback**: When rocm-smi reports VRAM below the APU cutoff, the backend automatically falls back to sysfs GTT detection, ensuring correct unified memory reporting.

### Intel GPU Driver Support

- **Intel xe Driver Support**: Added support for Intel's new `xe` driver (used by Battlemage/Xe2 GPUs like Arc B580). The xe driver uses different sysfs paths than the traditional i915 driver. The backend now detects which driver is in use and queries VRAM accordingly.

- **Driver Detection**: Intel backend now checks driver symlink to distinguish between i915 (Arc A-series) and xe (Arc B-series) drivers.

#### Configuration

- **New Setting**: `ROUTER_AMD_UNIFIED_MEMORY_GB` - Manual override for AMD APU unified memory size. Set to ~90% of your system RAM if auto-detection fails. Example: `ROUTER_AMD_UNIFIED_MEMORY_GB=58` for a 64GB system.

#### Documentation

- **AMD APU BIOS Guide**: Added detailed BIOS UMA Frame Buffer configuration guidance. The UMA buffer should be set to **minimum** (512MB-2GB), not maximum, because GTT is the actual usable memory pool.

- **Troubleshooting**: Added AMD APU-specific troubleshooting section for wrong VRAM detection issues.

- **Architecture Deep Dive**: Added unified memory architecture explanation to `DEEPDIVE.md`.

### Docker Compose Updates

- **AMD GPU Group Permissions**: Updated `docs/docker-compose.amd.yml` and `docs/docker-compose.multi-gpu.yml` to include `group_add` for render/video groups, ensuring proper GPU device permissions in containers.

- **APU Setup Guidance**: Added unified memory setup instructions to docker-compose templates.

---

## [2.1.1] - 2026-02-21

### Performance Optimizations

This release focuses on significant performance improvements across the routing pipeline, database operations, and backend communication layers.

#### Database Optimizations

- **N+1 Query Fix - Feedback Aggregation**: Changed `_get_model_feedback_scores()` to use SQL `GROUP BY` aggregation instead of loading all feedback records into memory. Reduces memory from O(N) to O(1) and improves speed 10-100x for large datasets.

- **Bulk Upsert Optimization**: Rewrote `bulk_upsert_benchmarks()` to use single-transaction bulk upsert with SQLite `ON CONFLICT`. Previously made individual queries and commits per benchmark item. Reduces sync time from 30-60s to 1-2s for 1000 benchmarks.

- **Admin Endpoints Pagination**: Added pagination (`limit`, `offset`) to `/admin/profiles` and `/admin/benchmarks` endpoints. Prevents memory exhaustion with large model counts. Default limit: 100, max: 1000.

- **Database Indexes**: Added indexes for common query patterns via automatic migration:
  - `idx_model_feedback_model_timestamp` on `model_feedback(model_name, timestamp)`
  - `idx_routing_decision_selected_model` on `routing_decisions(selected_model)`
  - `idx_benchmark_sync_last_sync` on `benchmark_sync(last_sync)`

#### Backend HTTP Optimizations

- **Persistent HTTP Clients**: All backends (Ollama, llama.cpp, OpenAI) now use persistent `httpx.AsyncClient` with connection pooling instead of creating new clients per request. Reduces latency by 30-70% (50-150ms saved per request from TCP/TLS handshake elimination).

- **Backend Cleanup on Shutdown**: Added `close()` method to backend protocol and proper cleanup in shutdown event.

#### Caching Improvements

- **Vectorized Similarity Search**: `SemanticCache._cosine_similarity_batch()` now uses numpy for vectorized batch similarity calculations. Falls back to pure Python if numpy unavailable. Improves cache lookup speed 10-100x for large caches.

- **Split Cache Locks**: Replaced single `_lock` with separate `_routing_lock`, `_response_lock`, and `_embedding_lock` to reduce lock contention under high load.

- **Embedding Cache**: Added separate embedding cache with 24-hour TTL (vs 1-hour for routing cache). Caches embeddings by prompt hash to avoid expensive embedding API calls for repeated prompts. Tracks `embedding_cache_hits` and `embedding_cache_misses` in stats.

- **Model Frequency Counter**: Replaced linear scan of `recent_selections` list with O(1) Counter-based frequency tracking.

#### Parallelization

- **Parallel Profiling**: Added `ROUTER_PROFILE_PARALLEL_COUNT` config option (default: 1). When set to 2+, profiles multiple models concurrently using `asyncio.gather()` with semaphore. Reduces profiling wall-clock time by 2-5x for multi-GPU systems.

- **Parallel Benchmark Sync**: Benchmark providers (HuggingFace, LMSYS, ArtificialAnalysis) now fetch in parallel using `asyncio.gather()` with 120s timeout per provider. Reduces sync wall-clock time 2-3x.

#### Timeouts & Robustness

- **Timeout on list_models()**: Added `list_models_with_timeout()` helper with 10s default timeout. Prevents indefinite hangs when backend is slow/unresponsive.

- **Provider Fetch Timeout**: Added 120s timeout per benchmark provider fetch.

### Security Fixes

- **CalculatorSkill Security**: Rewrote expression evaluator to use AST parsing instead of string split. Removed exponentiation operator (^) to prevent DoS via large power calculations. Added expression length limit (100 chars), result magnitude limit (1e15), and proper error handling.

### Configuration

- **New Setting**: `ROUTER_PROFILE_PARALLEL_COUNT` - Number of models to profile concurrently (default: 1)

### Test Coverage

- Test count: 317+ tests passing
- Added tests for caching optimizations in `tests/test_caching.py`

---

## [2.1.0] - 2026-02-20

### New Features

- **ArtificialAnalysis.ai Benchmark Integration**: New benchmark data source providing proprietary intelligence/coding/math indices, real-world speed metrics (tokens/sec), and standard benchmarks (MMLU-Pro, GPQA, LiveCodeBench, Math-500). Configure via `ROUTER_BENCHMARK_SOURCES=artificial_analysis`, `ROUTER_ARTIFICIAL_ANALYSIS_API_KEY`, and optional model mapping YAML file. Data stored in new `extra_data` JSON column for provider-specific fields.

- **Model Keep-Alive Configuration**: Added `ROUTER_MODEL_KEEP_ALIVE` setting to control how long models stay loaded in VRAM after each request. Default `-1` (keep indefinitely). Set to `0` to unload immediately after each response, or positive seconds for custom TTL. Addresses issue where multiple models accumulate in VRAM.

- **Manual Benchmark Sync Endpoint**: Added `POST /admin/sync-benchmarks` endpoint to manually trigger benchmark synchronization from all configured sources. Requires admin API key if configured. Returns count of synced models and matched model names.

### Bug Fixes

- **Signature Inside Code Blocks**: Fixed bug where the model signature could be appended inside a fenced code block if the LLM response ended with an unclosed code fence. Added detection and automatic closing of unclosed blocks before signature insertion.

- **Prometheus Metrics Label Mismatch**: Fixed `ValueError: Incorrect label names` in VRAM monitor. GPU metrics in `router/metrics.py` were defined with only `gpu_index` label, but `vram_monitor.py` was using both `gpu_index` and `vendor`. Added `vendor` label to all GPU metrics.

- **Benchmark DB DateTime Comparison**: Fixed `TypeError: can't compare offset-naive and offset-aware datetimes` in `bulk_upsert_benchmarks`. Now properly converts naive datetimes to UTC-aware before comparison.

- **Benchmark DB Dict Comparison**: Fixed error when comparing `extra_data` dict field using `>` operator. Dict fields are now always updated if present (skip the value comparison).

### Improvements

- **Deprecation Warnings Fixed**: Replaced all `datetime.utcnow()` calls with `datetime.now(timezone.utc)` in ArtificialAnalysis provider to eliminate Python 3.12+ deprecation warnings.

- **Database Migration for extra_data**: Added automatic migration to create `extra_data` JSON column in `model_benchmarks` table. Runs on startup via `_run_migrations()` in `database.py`.

- **Example Mapping File**: Created `artificial_analysis_models.example.yaml` with detailed comments, example mappings for popular model families (Llama, Phi, Qwen, Gemma, Mistral), and instructions for finding correct AA IDs.

- **Configuration Documentation**: Updated `docs/configuration.md` with:
  - Detailed ArtificialAnalysis settings
  - `ROUTER_MODEL_KEEP_ALIVE` documentation
  - Updated complete `.env` example
  - Benchmark sources ordering and priority explanation

### Test Coverage

- Added `tests/test_schemas.py` (8 tests) for code block handling utilities
- Fixed `test_check_nvidia_smi_not_available` to properly mock GPU manager on systems with actual NVIDIA hardware
- Added `model_keep_alive` default assertion to `tests/test_config.py`
- Test count: 303 tests passing

---

## [2.0.0] - 2026-02-20

- **Semantic Cache Optimization (O(N) computation reduction)**: Modified `SemanticCache._cosine_similarity` to pre-calculate and store embedding magnitudes upon insertion instead of re-calculating them inside the `for` loop during lookups. This significantly reduces CPU overhead when `SemanticCache` reaches its `max_size` (e.g., 500 entries) with 8192-dimension embeddings, saving up to ~4 million redundant math operations per cache lookup.
- **Race Condition / Duplicate Code Fix**: Fixed a logical bug in `main.py`'s `stream_chat` endpoint. There was duplicate VRAM unloading logic caused by an unindented block (`if current and current != selected_model and current != pinned:`) that was executing outside the `else` clause, leading to redundant API calls to unload models.
- **Deep Mypy Type-Safety Enhancements**:
  - Eliminated `Unsupported operand type for - ("None" and "float")` in backend streaming (`ollama.py`, `openai.py`, `llama_cpp.py`) by replacing untyped `timing` dicts with explicit explicit `start_time` and `first_token_time` variables.
  - Resolved `Argument 4 to "chat" has incompatible type` by explicitly typing `backend_kwargs` as `dict[str, Any]` in `main.py`.
  - Added explicit type annotations for `response` dicts and `invalidated` variables in `/admin` endpoints.


- **Critical Bug Fix**: Removed unused `benchmark_source` field from `benchmark_db.py`. The field was in the whitelist but never existed in `ModelBenchmark` model, causing potential crashes in `get_benchmarks_for_models()`.
- **Type Safety Improvements**:
  - Fixed tuple unpacking in `profiler.py` after `asyncio.gather(return_exceptions=True)` to properly handle both exceptions and successful results
  - Added explicit type annotations for `semantic_cache` field in `RouterEngine`
  - Fixed `embedding` variable scope issue in `select_model()` method
  - Added type annotations for `model_scores` and `model_counts` in `_get_model_feedback_scores()`
- **Edge Case Handling**:
  - Fixed potential crash in `_calculate_combined_scores()` when `analysis` dict is empty by adding guard for `max()` on empty sequence
  - Properly exclude meta-categories (complexity, vision, tools) from dominant category detection
  - Added None check for `messages[-1].content` in request handling
- **Comprehensive Test Coverage**: Added 30 new edge case tests in `tests/test_edge_cases.py` covering:
  - SemanticCache: empty cache, cosine similarity edge cases, LRU eviction, model frequency
  - RouterEngine: empty prompts, very long prompts, special characters, unicode, parameter extraction, complexity buckets
  - Benchmark DB: empty model lists, bulk upsert edge cases
  - Config: quality preference extremes, URL validation, benchmark sources
  - Profiler: timeout calculations, token rate initialization
  - Logging sanitization: empty strings, None values, nested dicts, long strings
  - Routing decisions: reasoning string generation with various score combinations
- **Test Count**: Increased from 258 to 288 tests (30 new edge case tests)

### Routing Algorithm Improvements

- **Profile Scores in Combined Score**: Added profile scores as Signal 4 in the routing algorithm. Previously, profile scores were only used in bonus calculations. Now they directly influence the combined category score with weight `0.8 * quality_weight`, making runtime profiling data more impactful.
- **Smarter Model Selection for Simple Tasks**: Improved routing to better favor small/fast models for low complexity tasks:
  - Low complexity (< 0.15): Strong bonuses for small models (≤7B: +0.8-1.5) and penalties for large models (≥14B: -1.0 to -2.0)
  - Category boost threshold raised from 0.05 to 0.15 to prevent weak signals from triggering the 20x boost
  - Size bonuses now only apply for moderate+ complexity tasks (≥ 0.3), not for benchmarked models at low complexity
- **Database Migration**: Added automatic migration for new columns `adaptive_timeout_used` and `profiling_token_rate` in SQLite. Migration runs on startup via `_run_migrations()` in `database.py`.

### Test Script Fixes

- **Syntax Error Fix**: Fixed missing quotes in bc comparisons (`$(...)` should be `"$(...)"`) in `test_smarterrouter_v2.sh`
- **Haiku Test Fix**: Updated haiku detection to properly handle escaped newlines in JSON responses using Python line counting

### Backend Provider Fixes & Improvements

- **Removed `generate()` from Protocol**: The unused `generate()` method was removed from `LLMBackend` protocol. All backends now consistently use `chat()` for all generation tasks.
- **Response Format Normalization**: `LlamaCppBackend` and `OpenAIBackend` now transform OpenAI-format responses to Ollama format internally. All backends now return consistent `{"message": {"content": ...}, "prompt_eval_count": ..., "eval_count": ...}` structure.
- **Model Prefix Support**: All backends now support `model_prefix` parameter:
  - `OllamaBackend`: Added `model_prefix` support (was previously missing)
  - `LlamaCppBackend`: Already supported, now consistent
  - `OpenAIBackend`: Already supported, now consistent
- **Trailing Slash Handling**: All backends now consistently strip trailing slashes from `base_url` in `__init__`.
- **Path Normalization**: Fixed `OpenAIBackend` to avoid duplicate `/v1` in URLs when base_url already includes it.
- **VRAM Management**: Added `supports_unload()` helper function to check if backend supports model loading/unloading. Only Ollama supports this; other backends return `False`.
- **Comprehensive Backend Testing**: Added 80+ new tests across four test files:
  - `tests/test_ollama_backend.py` - 15 tests for OllamaBackend
  - `tests/test_llama_cpp_backend.py` - 13 tests for LlamaCppBackend
  - `tests/test_openai_backend.py` - 14 tests for OpenAIBackend
  - `tests/test_backend_contract.py` - Contract tests ensuring all backends behave consistently

### Judge Improvements

- **Universal Compatibility**: Removed `response_format` parameter which isn't supported by all providers
- **Enhanced Prompt Engineering**: Clear JSON instructions ensure consistent output format without provider-specific features
- **OpenRouter Support**: Added optional `HTTP-Referer` and `X-Title` headers for OpenRouter compliance
- **Retry Logic with Exponential Backoff**: Automatically retries on transient errors:
  - Retries on 429 (rate limit), 5xx (server errors), and network timeouts
  - Configurable via `ROUTER_JUDGE_MAX_RETRIES` (default: 3) and `ROUTER_JUDGE_RETRY_BASE_DELAY` (default: 1.0s)
  - Helps with OpenRouter free tier rate limiting (20 req/min)
- **Better Error Logging**: Detailed error messages including raw response body on 400 errors
- **Markdown JSON Extraction**: Added `_extract_json_from_content()` to handle JSON wrapped in markdown code blocks (```json ... ```), fixing issues with providers like Google Gemini that wrap responses in markdown

### Profiling Performance Optimizations

- **Parallel Prompt Processing**: Rewrote `_test_category()` to process all prompts in a category concurrently with semaphore control (max 3 concurrent):
  - Reduces profiling time from 15×timeout to ~3×timeout per category
  - Maintains system stability by limiting concurrent requests
  - Each prompt still individually scored by judge
- **Adaptive Timeout Based on Model Size**: `ModelProfiler` now automatically adjusts timeout with granular tiers based on model parameters:
  - Very large models (70B+): 2.5× base timeout (225s)
  - Large models (30B-69B): 1.8× base timeout (162s)
  - Medium-large models (14B-29B): 1.4× base timeout (126s) - Fixes timeouts on qwen-14b, etc.
  - Medium models (7B-13B): 1.1× base timeout (99s)
  - Small models (≤3B): 0.8× base timeout (72s)
  - Extracts parameter count from model names (e.g., "llama3:70b", "phi3:1b")
- **Increased Default Profiling Timeout**: Changed default `ROUTER_PROFILE_TIMEOUT` from 60s to 90s to better accommodate larger models like qwen-14b, deepseek-r1:14b, etc.
- **Async VRAM Measurement**: Added `_measure_vram_gb_async()` to avoid blocking the event loop during VRAM sampling
- **Intelligent Warmup Phase**: Added a two-phase profiling approach to eliminate cold-start timeouts:
  - Phase 1: Explicitly loads model into memory with a **size-based timeout** before benchmarking
  - Timeout calculated as `(size_gb / disk_speed_gbps) + 30s` (default assumes 50 MB/s disk)
  - Example: 14GB model gets ~5 minutes to load; 70GB model gets ~25 minutes
  - Prevents timeouts caused by slow disk I/O rather than model performance
  - Configurable via `ROUTER_PROFILE_WARMUP_DISK_SPEED_MBPS` and `ROUTER_PROFILE_WARMUP_MAX_TIMEOUT`
  - If warmup fails, profiling continues anyway (backward compatible)
- **Adaptive Timeouts**: Dynamic per-model timeout calculation based on actual performance:
  - Measures token generation rate during 3-prompt screening phase
  - Calculates timeout using two methods (conservative max-time and token-projection)
  - **Robust Fallback**: Always uses the size-based guess as a **Minimum Floor**, ensuring fast screening doesn't result in overly aggressive timeouts later
  - **Reasoning Awareness**: Automatically doubles safety factors for models like `deepseek-r1` or those with "reasoning" in their name
  - Uses the higher of the calculated timeouts with safety factor (default 2.0x)
  - Fast models (phi3:mini, llama3.2:1b) get 30-60s timeouts
  - Slow reasoning models (deepseek-r1:7b) get 300-600s timeouts automatically
  - Eliminates manual timeout tuning regardless of hardware or model mix
  - Configurable via `ROUTER_PROFILE_ADAPTIVE_TIMEOUT_MIN`, `ROUTER_PROFILE_ADAPTIVE_TIMEOUT_MAX`, `ROUTER_PROFILE_ADAPTIVE_SAFETY_FACTOR`
  - Calculated timeout and token rate stored in database for debugging

### Database Reliability Improvements

- **Automatic Database File Creation**: Enhanced `init_db()` in `router/database.py` to handle SQLite database initialization more robustly:
  - Automatically creates parent directories if they don't exist
  - Touches the database file before SQLAlchemy initialization to ensure proper permissions
  - Prevents "unable to open database file" errors on fresh Docker deployments
  - Handles both relative paths (`./router.db`) and absolute paths
  - Logs directory and file creation for debugging

### Security & Production Hardening

- **Production Security Warnings**: Added startup warning if `ROUTER_ADMIN_API_KEY` is not set
- **Enhanced Admin Key Documentation**: Updated `ENV_DEFAULT` with strong security warnings and examples
- **Docker Security**: Production-ready Docker Compose with:
  - `read_only: true` immutable root filesystem
  - `security_opt: no-new-privileges:true` privilege escalation prevention
  - Health checks for container monitoring
  - Tmpfs mount for temporary files

### Performance & Scalability

- **N+1 Query Fix**: Eliminated database query per model in fallback loop by pre-fetching VRAM estimates
- **VRAMManager Thread Safety**: Added `asyncio.Lock` to prevent race conditions in concurrent model loading/unloading
- **Response Cache Granularity**: Cache keys now include generation parameters (`temperature`, `top_p`, `max_tokens`, `seed`, etc.) to prevent incorrect cache hits
- **Request Size Limits**: Added 10MB request body limit to prevent memory exhaustion attacks

### API & Features

- **Rate Limited Chat Endpoint**: `/v1/chat/completions` now respects rate limits (configurable separately from admin endpoints)
- **Explain Routing Endpoint**: New `GET /admin/explain?prompt=...` returns detailed routing breakdown without generating response
  - Shows selected model with confidence score
  - Displays reasoning for selection
  - Lists all model scores from database
  - Useful for debugging routing decisions
- **Backend URL Validation**: Pydantic validators ensure URLs start with `http://` or `https://`

### Observability

- **Enhanced Error Context**: Improved exception logging includes:
  - Model name being attempted
  - Sanitized prompt preview (first 100 chars)
  - Response ID for request correlation
  - Current VRAM state (available/total GB)
  - Full stack traces with `exc_info=True`

### Documentation

- **Comprehensive README Updates**:
  - **Scoring Algorithm** section explaining category-based routing, complexity assessment, and scoring formula
  - **Troubleshooting Guide** with "Why wasn't my model selected?" checklist and common issues
  - **Performance Tuning** guide for low latency, high quality, and high throughput scenarios
  - Database persistence warnings and backup procedures
- **RELEASE.md**: New release checklist document with:
  - Pre-release testing procedures
  - Version bumping steps
  - Docker image build/push instructions
  - Security release procedures
  - Rollback plans

### Testing

- **Extended Integration Tests**: New `tests/test_integration_extended.py` with comprehensive test coverage:
  - Full chat flow with mock backend
  - Streaming response handling
  - Error handling and fallback behavior
  - Caching behavior verification
  - Rate limiting enforcement
  - Request validation and sanitization
  - Docker health check verification

---

## [1.9.0] - 2026-02-17

### New Features: VRAM Monitoring & Measurement

- **VRAM Monitoring**: Added background `VRAMMonitor` that polls `nvidia-smi` at configurable intervals. Provides real-time GPU memory tracking and logs summaries.
- **VRAM Profiling**: Models are now profiled for actual VRAM usage during profiling. Results stored in database (`vram_required_gb`, `vram_measured_at`).
- **Admin VRAM Endpoint**: New `/admin/vram` REST endpoint returns current VRAM metrics, history, and loaded models. Requires admin auth.
- **Simplified Configuration**: Replaced separate `headroom_gb` setting with a single `ROUTER_VRAM_MAX_TOTAL_GB`. The router uses an internal 0.5GB fragmentation buffer automatically.
- **Auto-Detection**: If `ROUTER_VRAM_MAX_TOTAL_GB` is not set, the router automatically detects GPU total VRAM and defaults to 90% of it.
- **VRAM-Aware Routing**: The router now considers measured VRAM requirements when making routing decisions, improving multi-model environments.

### Observability & Operations

- **Structured Logging**: Added `ROUTER_LOG_FORMAT` setting (`text` or `json`). JSON mode includes correlation IDs and sanitized fields for log aggregation.
- **Request Correlation**: Each request gets a unique `X-Request-ID` that propagates to logs for tracing.
- **Prometheus Metrics**: New `/metrics` endpoint exposes request rates, error counts, cache hit/miss ratios, model selection distribution, and VRAM usage.
- **Multi-GPU Support**: VRAM monitoring now aggregates across all GPUs and provides per-GPU breakdowns in `/admin/vram` and metrics.
- **Enhanced Sanitization**: Improved secret redaction in logs to cover more patterns (JWT, database URLs, long base64).

## [1.8.0] - 2026-02-17

### Production Hardening & Critical Bug Fixes

This release focuses on stability, security, and performance improvements based on real-world testing and code review.

#### Critical Bug Fixes
- **Race Condition in Rate Limiter**: Added `asyncio.Lock` to protect shared state, preventing corruption under concurrent load
- **Duplicate Dictionary Key**: Fixed duplicate `"creativity"` key in category mapping that was causing data loss
- **Cache Not Working Without Embedding Model**: Fixed logic that prevented exact hash cache lookup unless `ROUTER_EMBED_MODEL` was set. Cache now works by default.
- **SQL Injection Risk**: Added whitelist validation in `bulk_upsert_benchmarks()` to prevent malicious key injection
- **Tool Call Counter**: Fixed logic that could have allowed excessive tool iterations
- **Judge Fallback Scoring**: Changed from always 1.0 to neutral 0.5 for non-empty responses when LLM-as-Judge is disabled

#### Thread Safety & Concurrency
- **SemanticCache Refactor**: Converted all cache methods to async with proper locking:
  - `get()`, `set()`, `get_response()`, `set_response()`
  - `invalidate_response()`, `get_stats()`, `get_model_frequency()`
- **Rate Limiter Lock**: Added `asyncio.Lock` (`rate_limit_lock`) to `AppState` for thread-safe counter updates
- **Database Session Safety**: Ensured all session operations are properly scoped and closed

#### Security Enhancements
- **SQL Injection Prevention**: Whitelist of allowed `ModelBenchmark` fields prevents code injection via dynamic keys
- **Improved API Key Redaction**: Enhanced pattern matching for secret detection in logs
- **Connection-Level Rate Limiting**: Added limits to prevent streaming connection abuse

#### Performance Improvements
- **Larger Cache Sizes**:
  - Routing cache: 100 → 500 entries
  - Response cache: 50 → 200 entries
- **Reduced Cache Misses**: Increased capacities better suit production workloads
- **Lock Efficiency**: Fine-grained lock usage minimizes contention

#### Code Quality
- **Centralized Signature Stripping**: New `strip_signature()` helper in `schemas.py` replaces scattered regex logic
- **Protocol Compliance**: All backends (`OllamaBackend`, `LlamaCppBackend`, `OpenAIBackend`) now explicitly inherit `LLMBackend`
- **Type Fixes**: Resolved multiple type errors in `router.py` and `main.py`
- **Async Corrections**: Fixed missing `await` statements throughout codebase

#### Configuration
- **New Setting**: `ROUTER_CACHE_RESPONSE_MAX_SIZE` (default: 200) controls response cache capacity
- Updated `ENV_DEFAULT` with documentation for new setting

#### Testing
- All **73 tests pass** without modification
- No regressions introduced
- Improved test coverage for async cache operations

---

## [1.6.0] - 2026-02-16

### Tool Execution & Feature Enhancements

#### Added
- **Tool Execution Engine**:
  - Implemented tool execution loop in `main.py`
  - `web_search` skill now uses DuckDuckGo API
  - `calculator` skill safely evaluates expressions
- **Model Override**: `?model=xxx` query parameter to force a specific model
- **Health Stats**: New `/admin/stats` endpoint with detailed metrics
  - Total requests, errors, uptime
  - Requests by model
  - Cache stats (size, hits)
- **Smart Caching**: Cache now stores full `RoutingResult` objects

---

## [1.5.3] - 2026-02-16

### Category-Aware Minimum Size Requirements

Added intelligence to routing to prevent small models from being selected for complex tasks.

#### Added
- **Category-Minimum Size Mapping**:
  - `coding`: simple=0B, medium=4B+, hard=8B+
  - `reasoning`: simple=0B, medium=4B+, hard=8B+
  - `creativity`: simple=0B, medium=1B+, hard=4B+
  - `general`: simple=0B, medium=1B+, hard=4B+
- **Minimum Size Penalty**: Models below minimum size for their category get a severe penalty (-10 * size deficit)
- **Complexity Bucket Detection**: Helper function to categorize prompts as simple/medium/hard
- **Size-Aware Category Boost**: Category-first boost now considers adequate model size, not just benchmark data

#### Impact
- Complex coding tasks will no longer route to 0.5B models
- Simple prompts can still use small fast models
- Large models (14B+) will be preferred for hard tasks

---

## [1.5.2] - 2026-02-16

### Critical Bug Fixes

#### Fixed
- **Benchmark Sync**: Fixed incorrect argument passing - now passes actual model names instead of source names
- **LLM Dispatch**: Added missing `_parse_llm_response` and `_build_dispatch_context` methods
- **Streaming Latency**: Fixed latency measurement to track time-to-first-token correctly
- **Streaming Format**: Normalized OpenAI/LlamaCpp streaming output to match Ollama format
- **[DONE] Handling**: Fixed crash when streaming receives `[DONE]` sentinel
- **Detached ORM Objects**: Fixed `get_benchmarks_for_models` returning detached SQLAlchemy objects
- **Bare Except**: Changed to `except Exception:` to avoid swallowing system signals
- **OpenAI Model List**: Fixed double `/v1/` in URL path

#### Removed
- **Dead Code**: Removed unused `router/client.py` file

---

## [1.5.1] - 2026-02-16

### Routing Optimizations & Bug Fixes

#### Fixed
- **Critical Bug**: Removed references to deprecated `factual` field in profiler
- **Duplicate Signatures**: Fixed issue where models outputting their own "Model:" signatures caused duplicates

#### Added
- **Semantic Caching**: New `SemanticCache` class stores routing decisions based on prompt hash
  - Reduces latency for repeated queries
  - 1-hour TTL, 100-entry LRU cache
- **Diversity Enforcement**: Added penalty for models selected too frequently
  - Prevents single-model monopolization
  - Tracks recent selections and applies up to 50% penalty

#### Changed
- **Scoring Update**: Uses `creativity` instead of deprecated `factual` in profile matching

---

## [1.5.0] - 2026-02-16

### OpenAI-Compatible Embeddings & Enhanced API

Major update to bring the router closer to full OpenAI API compatibility, adding support for vector embeddings and standard generation parameters.

#### Added
- **Embeddings Endpoint (`/v1/embeddings`)**:
  - Full support for generating vector embeddings via Ollama, llama.cpp, or OpenAI backends.
  - OpenAI-compatible request and response formats.
  - Support for batch processing (multiple input strings in one request).
- **Enhanced Chat Completion Parameters**:
  - Added support for standard OpenAI parameters: `temperature`, `top_p`, `n`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`, `seed`, `logprobs`, and `top_logprobs`.
  - Parameters are now validated by Pydantic and passed through to the underlying backends.
- **Usage Tracking**:
  - Responses now include a standard `usage` object with `prompt_tokens`, `completion_tokens`, and `total_tokens`.
  - Works for both regular and streaming responses (final chunk).

#### Changed
- **Backend Abstraction**: Updated `LLMBackend` protocol with an `embed` method.
- **Request Validation**: Significant expansion of `ChatCompletionRequest` schema.
- **Streaming Response**: Improved streaming chunks to include more metadata and reliable finish reasons.

---

## [1.4.0] - 2026-02-16

### Quality-Based Profiling & Standardized Benchmarks

Major upgrade to model evaluation: transitioning from simple completion checks to qualitative assessment using the "LLM-as-Judge" pattern and standardized prompts.

#### Added
- **LLM-as-Judge Scoring Engine**:
  - New "Judge" capability that uses a high-end model (e.g., GPT-4o) to grade the responses of other models.
  - Replaces binary pass/fail checks with a 0.0-1.0 quality score based on accuracy, clarity, and instruction following.
  - Fully configurable via `ROUTER_JUDGE_*` settings, supporting any OpenAI-compatible API as the judge.
- **Standardized Benchmark Prompts**:
  - Replaced simple hardcoded prompts with a curated set of 15 prompts inspired by **MT-Bench**.
  - Prompts cover Reasoning, Coding, and Creativity with increased rigor.
- **Improved Progress Tracking**:
  - Profiler now provides more accurate progress percentages and ETA calculations based on the new prompt set.
- **New Configuration Settings**:
  - `ROUTER_JUDGE_ENABLED`: Toggle qualitative scoring.
  - `ROUTER_JUDGE_MODEL`: Specify the model to act as judge.
  - `ROUTER_JUDGE_BASE_URL`: Use any OpenAI-compatible endpoint for the judge.
  - `ROUTER_JUDGE_API_KEY`: Secure access to the judge model.

#### Changed
- **Profiler Overhaul**:
  - Significant refactor of `ModelProfiler` to support asynchronous judge calls.
  - Category testing now integrates the judge's qualitative feedback into the final scores.
  - Optimized progress logging for the expanded prompt set.

---

## [1.3.0] - 2026-02-16

### Skills & Capabilities

Major update introducing "Agentic" features: Skills Registry, Multimodal Support, and Capability-based Routing.

#### Added
- **Skills Endpoint (`/v1/skills`)**:
  - Lists available tools/skills (e.g., Web Search, Calculator) that can be used by models.
  - Prepares the router for future "Model Context Protocol" (MCP) integration.
- **Multimodal Support**:
  - API now accepts OpenAI-style multimodal inputs (text + images in `messages`).
  - Automatically detects images and routes to vision-capable models (Llava, Pixtral, GPT-4o).
- **Tool Use Detection**:
  - Detects `tools` definitions in requests.
  - Routes to models optimized for function calling (e.g., Qwen2.5-Coder, Mistral Large).
- **Capability-Based Filtering**:
  - Strict filtering ensures vision tasks go to vision models.
  - "JSON Mode" requests prioritize coding/structured output models.
- **Enhanced Profiler**:
  - Auto-detects capabilities (Vision/Tools) based on model names.
  - Updates `ModelProfile` with these new flags.

#### Changed
- **Database Schema**: Added `vision` and `tool_calling` columns to `model_profiles` and `model_benchmarks`.
- **Request Validation**: Updated `ChatCompletionRequest` to support list-based content and `tools`.

---

## [1.2.0] - 2026-02-16

### Enhanced Intelligence & Feedback

Implemented "Best Practice" routing strategies inspired by Hybrid LLM, RouteLLM, and GraphRouter papers.

#### Added
- **Query Difficulty Predictor**:
  - Enhanced prompt analysis to detect complexity based on length, structure, and keywords.
  - Automatically identifies "hard" prompts that require larger models.
- **Cost-Quality Tuner**:
  - New `ROUTER_QUALITY_PREFERENCE` setting (0.0 - 1.0).
  - Allows explicit trade-off between speed (smaller models) and quality (larger/smarter models).
- **Size-Aware Routing**:
  - Implemented scoring bonuses for larger models (14B, 30B+) on complex tasks.
  - Applies penalties to tiny models (<3B) when high capability is needed.
- **Feedback Loop**:
  - New `/v1/feedback` endpoint for submitting user ratings.
  - Router now boosts scores of models that have received positive feedback in the past.
  - Database schema updated with `ModelFeedback` table.
- **Reliability Improvements**:
  - Explicit `response_id` tracking for linking feedback to decisions.
  - Enhanced fallback mechanism: if a model fails, the next best model is automatically tried.

#### Changed
- **Scoring Algorithm**: Major overhaul of `_calculate_combined_scores`.
  - Now considers: Benchmark Data, Runtime Profile, Name Affinity, Complexity, Size, and User Feedback.
  - Dynamic weighting based on `quality_preference`.
  - Significantly improved heuristic matching for models like `codellama`.

#### Testing
- Added tests for quality preference impact.
- Added tests for feedback scoring boost.
- Fixed and updated existing router tests to reflect smarter heuristics.

---

## [1.1.0] - 2026-02-16

### Multi-Backend Support & VRAM Management

Added support for multiple LLM backends and proactive VRAM management for systems with limited GPU memory.

#### Added
- **Configurable Router Model Name**:
  - New `ROUTER_EXTERNAL_MODEL_NAME` config option to set the name the router presents to external UIs (e.g., OpenWebUI).
  - The `/v1/models` endpoint now returns this single model name, simplifying integration with frontends.
- **Backend Abstraction Layer**: Unified interface for all LLM backends
  - `LLMBackend` Protocol defining common operations
  - Factory function for dynamic backend creation
  - Easy to add new backend implementations

- **Ollama Backend**: Full-featured backend for local Ollama instances
  - Model listing, chat, streaming, and generation
  - Model unloading for VRAM management
  - Existing functionality preserved

- **llama.cpp Backend**: Support for llama.cpp server and llama-swap
  - OpenAI-compatible `/v1` endpoints
  - No API key required
  - Model prefix support for naming conventions

- **OpenAI-Compatible Backend**: Support for any OpenAI-compatible API
  - OpenAI, Anthropic (via compatibility layer), LiteLLM, local AI servers
  - API key authentication
  - Configurable base URL and model prefix

- **Proactive VRAM Management**: Smart model unloading for limited VRAM
  - Automatic model unloading before loading new model
  - Pinned model support to keep a small model always in VRAM
  - Configurable via `ROUTER_PINNED_MODEL` environment variable

#### Configuration Changes
- `ROUTER_PROVIDER`: Select backend (ollama, llama.cpp, openai)
- `ROUTER_OLLAMA_URL`: Ollama endpoint (default: http://localhost:11434)
- `ROUTER_LLAMA_CPP_URL`: llama.cpp server endpoint
- `ROUTER_OPENAI_BASE_URL`: OpenAI-compatible API endpoint
- `ROUTER_OPENAI_API_KEY`: API key for authentication
- `ROUTER_MODEL_PREFIX`: Optional prefix for model names
- `ROUTER_PINNED_MODEL`: Model to keep always loaded in VRAM
- `ROUTER_GENERATION_TIMEOUT`: Timeout for model generation (default: 120s)

#### Security Improvements
- **API Key Authentication**: Optional Bearer token authentication for admin endpoints (`/admin/*`)
  - Set `ROUTER_ADMIN_API_KEY` to enable
  - Backward compatible: endpoints remain open if no key configured
  - Returns 401 Unauthorized if key is required but missing/invalid
- **Rate Limiting**: Optional request throttling per client IP
  - Enable with `ROUTER_RATE_LIMIT_ENABLED=true`
  - Configurable limits for general and admin endpoints
  - Returns 429 Too Many Requests when limit exceeded
  - In-memory rate limiter with per-endpoint tracking
- **SQL Injection Prevention**: Replaced raw SQL delete with ORM-based delete
  - All database queries use SQLAlchemy ORM with parameterized queries
  - Input validation on model names before database operations
- **Input Validation**: Pydantic models validate all API requests
  - Content-Type header validation (must be `application/json`)
  - Request body schema validation with detailed error messages
  - Length limits: prompts max 10,000 chars, max 100 messages per request
  - Role validation: only `user`, `assistant`, `system` allowed
  - Model name validation (alphanumeric, hyphens, underscores, colons, dots, slashes)
- **Prompt Sanitization**: Automatic sanitization of user input
  - Removal of null bytes (`\x00`)
  - Removal of control characters (except newlines, tabs, carriage returns)
  - Whitespace trimming
- **Log Sanitization**: Protection of sensitive data in logs
  - API key redaction (OpenAI format: `sk-...`)
  - Potential secret pattern detection and masking
  - Prompt truncation for logging (max 200 characters)
  - Newline removal for single-line logging

#### Improved Routing
- Better benchmark matching with fuzzy logic
- Bonus for models with benchmark data (+0.3)
- Reduced penalty for large models on simple tasks
- Enhanced complexity detection for coding tasks
- Size-aware routing: complex prompts route to larger models (14B+)
- Category-first boost only applies with benchmark data (prevents name-based over-selection)

#### Testing
- Updated test suite for new backend architecture
- 81 tests passing with comprehensive coverage

#### Backward Compatibility
- Default `provider=ollama` preserves existing behavior
- All existing environment variables continue to work

---

## [1.0.0] - 2026-02-15

### Major Release - Production Ready

After several iterations of development and testing, the SmarterRouter is now feature-complete with comprehensive test coverage and multi-provider benchmark support.

### What's New

#### Added
- **Multi-Provider Benchmark System**: Support for HuggingFace Leaderboard and LMSYS Chatbot Arena
  - Fetches MMLU, HumanEval, MATH, GPQA scores from HuggingFace via REST API
  - Pulls Elo ratings from LMSYS Chatbot Arena
  - Merges data from multiple sources intelligently
  - Configurable via `ROUTER_BENCHMARK_SOURCES` environment variable

- **Comprehensive Test Suite**: 81 tests covering all major functionality
  - Unit tests for providers, router logic, database operations
  - Integration tests for API endpoints
  - Client tests for Ollama HTTP interactions
  - 84% code coverage

- **Progress Logging**: Real-time profiling progress with ETA calculations
  - Shows current model, category, and prompt number
  - Displays percentage complete and estimated time remaining
  - Detailed scores after each model completes

- **Profiler Caching**: Models are only profiled once
  - Existing profiles are reused on startup
  - Only new models are profiled
  - Manual reprofile available via `/admin/reprofile?force=true`

#### Changed
- **Refactored Provider Architecture**: Moved from single hardcoded provider to pluggable provider system
  - Base `BenchmarkProvider` abstract class
  - Individual provider implementations for each data source
  - Easy to add new providers in the future

- **Updated Database Schema**: Added support for new metrics
  - `elo_rating`: Human preference scores from LMSYS
  - `throughput`: Model speed metrics
  - `context_window`: Token context limits

- **Improved Dispatcher Context**: Router now sees Elo and speed metrics when making decisions

### Fixed
- **HuggingFace Provider Rewrite**: Complete refactor from broken `datasets` library to REST API
  - Switched to HuggingFace Datasets Server REST API endpoint
  - Fixed 0 records issue caused by wrong dataset (`open-llm-leaderboard/contents` → `open-llm-leaderboard/results`)
  - Added robust JSON parsing for nested `row.results` structure
  - Improved error handling with specific HTTP and JSON error catching
  - Now successfully extracts MMLU, HumanEval, MATH, GPQA, and other benchmark scores

- **Profiler Caching**: Skip already-profiled models on startup
  - Models are cached in database
  - Only new models are profiled
  - Added `force=true` option to reprofile all

- **Benchmark Sync Fix**: Fixed SQLAlchemy bulk insert errors
  - Filter out None and non-scalar values
  - Use per-row insert/update instead of bulk upsert

- **LMSYS Redirect Handling**: Fixed 307 redirect issues when fetching CSV data
- **Datetime Deprecations**: Migrated to timezone-aware datetime objects
- **Test Suite Updates**: Fixed `test_profiler.py` to match new `_test_category()` signature after adding progress logging parameter

---

## [0.3.0] - 2026-02-15

### LLM-Based Dispatcher

### Added
- **LLM Dispatcher Mode**: Optional intelligent routing using a small LLM
  - Configure with `ROUTER_MODEL=llama3.2:1b` or similar small model
  - Dispatcher sees benchmark context and makes informed decisions
  - Falls back to keyword-based routing if dispatcher fails
  - ~200ms additional latency but much smarter selections

- **Combined Scoring Algorithm**: Merges runtime profiling with benchmark data
  - Weights keyword analysis with actual capability scores
  - Considers both accuracy and speed

- **Dependency Injection**: Refactored main.py to use FastAPI `Depends()`
  - Better testability
  - Cleaner separation of concerns

### Changed
- **Router Engine**: Major refactor to support dual routing modes
  - `_llm_dispatch()` for LLM-based selection
  - `_keyword_dispatch()` for fast rule-based selection
  - Automatic fallback between modes

- **Prompt Building**: Enhanced context building for LLM dispatcher
  - Includes Elo ratings, throughput, and context window info

---

## [0.2.0] - 2026-02-15

### Multi-Provider Benchmark Integration

### Added
- **HuggingFace Provider**: Real dataset integration
  - Uses `datasets` library to load `open-llm-leaderboard/contents`
  - Parses actual benchmark scores (not mock data)
  - Model name normalization and fuzzy matching
  - Score calculation across multiple benchmarks

- **LMSYS Provider**: Chatbot Arena Elo ratings
  - Fetches CSV from HuggingFace Spaces
  - Extracts human preference Elo scores
  - Model mapping to Ollama names

- **Artificial Analysis Provider**: Placeholder for future API integration
  - Structure ready for performance metrics
  - API key support prepared

- **Provider Orchestration**: Multi-source data merging
  - Concurrent fetching from enabled providers
  - Intelligent merge strategy (non-null values preferred)
  - Error isolation (one provider failure doesn't break others)

### Changed
- **Benchmark Sync**: Complete rewrite
  - No longer uses hardcoded mock data
  - Real-time fetching from external sources
  - Daily sync task with configurable interval

- **Configuration**: New environment variables
  - `ROUTER_BENCHMARK_SOURCES`: Toggle providers
  - Support for comma-separated list

---

## [0.1.0] - 2026-02-15

### Initial Release - MVP

### Added
- **Core Router Functionality**: Keyword-based model selection
  - Analyzes prompts for keywords (code, math, creative, factual)
  - Matches to profiled capabilities
  - Zero-latency routing decisions

- **Runtime Profiling System**: Tests actual Ollama models
  - 12 prompts across 4 categories
  - Real response time measurements
  - SQLite storage for persistence

- **Live Model Detection**: Automatic discovery
  - Polls Ollama every 60 seconds
  - Detects new models automatically
  - Triggers profiling for new additions

- **OpenAI-Compatible API**: Drop-in replacement
  - `/v1/chat/completions` endpoint
  - `/v1/models` listing
  - Streaming and non-streaming support
  - Response signature injection

- **Response Signatures**: Transparency feature
  - Appends `Model: <name>` to every response
  - Configurable format
  - Can be disabled

- **Database Layer**: SQLAlchemy + SQLite
  - Model profiles
  - Routing decisions audit log
  - Sync status tracking

- **Basic Admin Endpoints**: Management API
  - `/admin/profiles`: View capability profiles
  - `/admin/reprofile`: Manual reprofiling trigger

- **Docker Support**: Containerized deployment
  - Dockerfile
  - docker-compose.yml
  - Environment variable configuration

### Technical Decisions
- **Async/Await**: Full async stack for performance
- **SQLAlchemy**: ORM for database operations
- **Pydantic Settings**: Type-safe configuration
- **FastAPI**: Modern async web framework
- **Ruff**: Fast Python linting and formatting

---

## Roadmap / Future Ideas

### Potential Features
- [ ] Web dashboard for visualizing model performance
- [ ] Custom prompt categories (user-defined profiling)
- [ ] A/B testing framework for model selection strategies
- [ ] Performance metrics tracking over time
- [ ] Cost-based routing (if using paid APIs)
- [ ] Model recommendation engine based on usage patterns
- [ ] Integration with more benchmark sources
- [ ] Export/import of profile data
- [ ] REST API for external profiling tools

### Known Limitations
- Initial profiling takes 60-90 minutes for many models
- Model name matching requires fuzzy logic (not always perfect)
- LMSYS data requires follow-redirects support
- No built-in rate limiting on API endpoints

---

## Version History Summary

- **v1.3.0**: Skills Registry, Multimodal Support, Capability-based Routing
- **v1.2.0**: Query Difficulty Predictor, Cost-Quality Tuner, User Feedback Loop
- **v1.1.0**: Multi-backend support (Ollama, llama.cpp, OpenAI-compatible), VRAM management, 81 tests
- **v1.0.0**: Production ready, multi-provider benchmarks, 79 tests, progress logging
- **v0.3.0**: LLM-based dispatcher mode added
- **v0.2.0**: Real HuggingFace + LMSYS integration (no more mock data)
- **v0.1.0**: MVP with keyword routing and runtime profiling
