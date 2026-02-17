# LLM Router Proxy: Architectural Deep Dive

This document provides a technical overview of the LLM Router Proxy, detailing the design philosophy, component interactions, and the rationale behind specific architectural choices.

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
Running multiple models locally is a VRAM nightmare. 

- **Proactive Unloading**: Before the router tells a backend to load a new model, it can issue an unload command for the current model. This prevents "Out of Memory" errors during model switching.
- **Pinned Models**: You can "pin" a small, efficient model (like Phi-3 mini) to VRAM. The router will prioritize this model for simple queries, providing near-instant responses for the 80% of tasks that don't need a massive model.

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

## 5.1 Smart Caching

The router implements a multi-layered caching system to optimize performance:

### Routing Cache (Semantic)
- **Exact Hash Matching**: Uses SHA-256 of the prompt for instant cache hits.
- **Semantic Similarity**: If an embedding model is configured, uses cosine similarity to find similar prompts (threshold: 0.85 by default).
- **LRU Eviction**: Maintains up to 500 routing entries with 1-hour TTL.
- **Thread-Safe Operations**: All cache access is protected by an `asyncio.Lock`, ensuring correct behavior under concurrent load.
- **Tracks Recent Selections**: Keeps track of model selection frequency for diversity awareness and prevents model monopolization.

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
