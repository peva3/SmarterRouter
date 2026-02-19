## [Unreleased] - Pending

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


