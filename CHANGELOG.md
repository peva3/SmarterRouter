# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-16

### Multi-Backend Support & VRAM Management

Added support for multiple LLM backends and proactive VRAM management for systems with limited GPU memory.

#### Added
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

After several iterations of development and testing, the LLM Router Proxy is now feature-complete with comprehensive test coverage and multi-provider benchmark support.

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

- **v1.1.0** (Current): Multi-backend support (Ollama, llama.cpp, OpenAI-compatible), VRAM management, 81 tests
- **v1.0.0**: Production ready, multi-provider benchmarks, 79 tests, progress logging
- **v0.3.0**: LLM-based dispatcher mode added
- **v0.2.0**: Real HuggingFace + LMSYS integration (no more mock data)
- **v0.1.0**: MVP with keyword routing and runtime profiling

