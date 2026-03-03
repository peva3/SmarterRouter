# API Reference

SmarterRouter provides OpenAI-compatible endpoints for seamless integration with existing tools and applications.

## Base URL

- **Development:** `http://localhost:11436`
- **Production:** Configure based on your deployment

## Authentication

| Endpoint Type | Authentication Required |
|---------------|------------------------|
| `/v1/*` (chat, embeddings, models) | No |
| `/admin/*` | Yes - `ROUTER_ADMIN_API_KEY` required |
| `/health`, `/metrics` | No |

Admin endpoints require header: `Authorization: Bearer your-admin-api-key`

## Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "profiling_complete": true,
  "models_available": 5,
  "backend_connected": true,
  "timestamp": "2024-02-20T12:34:56.789Z"
}
```

### `GET /metrics`

Prometheus metrics endpoint for monitoring integration.

**Content-Type:** `text/plain; version=0.0.4`

**Metrics included:**
- `smarterrouter_requests_total` - Request count by endpoint and method
- `smarterrouter_request_duration_seconds` - Request duration histogram
- `smarterrouter_errors_total` - Error count by endpoint and type
- `smarterrouter_model_selections_total` - Model selection distribution
- `smarterrouter_cache_hits_total` / `cache_misses_total` - Cache statistics
- `smarterrouter_vram_total_gb`, `vram_used_gb`, `vram_utilization_pct` - GPU memory

### `GET /v1/models`

Returns the router itself as a single model.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "smarterrouter/main",
      "object": "model",
      "created": 1708162374.0,
      "owned_by": "local",
      "description": "An intelligent LLM router that automatically selects the best model..."
    }
  ]
}
```

### `POST /v1/chat/completions`

Main chat completion endpoint. Compatible with OpenAI API format.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Write a Python function..."}
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

**Streaming:** Set `"stream": true` for Server-Sent Events (SSE) format.

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1708162374,
  "model": "smarterrouter/main",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated response..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 150,
    "total_tokens": 165
  }
}
```

**Note:** The `model` field always returns `smarterrouter/main` as the router is the interface. The actual model used is appended to the response signature (see `ROUTER_SIGNATURE_ENABLED`).

### `POST /v1/embeddings`

Generate vector embeddings for text.

**Request:**
```json
{
  "model": "nomic-embed-text",
  "input": "The quick brown fox jumps over the lazy dog"
}
```

**Multiple inputs:**
```json
{
  "model": "nomic-embed-text",
  "input": ["text 1", "text 2", "text 3"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, ...],
      "index": 0
    }
  ],
  "model": "nomic-embed-text",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### `POST /v1/feedback`

Submit user feedback to improve routing decisions.

**Request:**
```json
{
  "response_id": "chatcmpl-...",
  "score": 1.0,
  "comment": "Great answer! Very helpful."
}
```

**Parameters:**
- `response_id` (required): Response ID from chat completion
- `score` (required): Float 0.0-2.0 where:
  - 0.0 = poor quality
  - 1.0 = acceptable/expected
  - 2.0 = exceptional/exceeded expectations
- `comment` (optional): Text feedback

**Response:** `200 OK` on success

---

## Admin Endpoints (Require Authentication)

All admin endpoints require `Authorization: Bearer your-admin-api-key` header.

### `GET /admin/profiles`

View performance profiles of all models.

**Response:**
```json
{
  "profiles": [
    {
      "model_name": "llama3:70b",
      "reasoning_score": 0.94,
      "coding_score": 0.87,
      "creativity_score": 0.78,
      "speed_score": 0.34,
      "vram_required_gb": 42.5,
      "last_profiled": "2024-02-20T10:30:00Z",
      "profiling_status": "completed"
    }
  ]
}
```

### `GET /admin/benchmarks`

View aggregated benchmark data from external sources (HuggingFace, LMSYS).

**Query params:**
- `?model=llama3:70b` - Filter to specific model

**Response:**
```json
{
  "benchmarks": {
    "llama3:70b": {
      "mmlu": 0.82,
      "humaneval": 0.78,
      "gsm8k": 0.85,
      "source": "huggingface",
      "last_updated": "2024-02-20T00:00:00Z"
    }
  }
}
```

### `POST /admin/reprofile`

Trigger manual reprofiling of models.

**Query params:**
- `?force=true` - Reprofile all models, even if already profiled
- `?models=llama3:70b,codellama:34b` - Specific models only

**Response:**
```json
{
  "message": "Reprofiling started for 3 models",
  "task_id": "abc123",
  "check_status": "/admin/profiling_status/abc123"
}
```

### `GET /admin/profiling_status/{task_id}`

Check status of a profiling task.

**Response:**
```json
{
  "task_id": "abc123",
  "status": "running",
  "progress": {
    "total": 5,
    "completed": 2,
    "current_model": "codellama:34b"
  },
  "estimated_completion": "2024-02-20T15:30:00Z"
}
```

### `GET /admin/vram`

View real-time VRAM usage and model memory allocation.

**Response:**
```json
{
  "total_gb": 23.8,
  "used_gb": 18.2,
  "free_gb": 5.6,
  "utilization_pct": 76.5,
  "gpus": [
    {
      "index": 0,
      "total_gb": 23.8,
      "used_gb": 18.2,
      "free_gb": 5.6
    }
  ],
  "loaded_models": [
    {
      "model_name": "llama3:70b",
      "vram_used_gb": 42.5,
      "loaded_at": "2024-02-20T10:15:00Z",
      "last_used": "2024-02-20T14:30:00Z"
    }
  ],
  "warnings": [
    "VRAM utilization above 75% threshold"
  ]
}
```

### `POST /admin/cache/invalidate`

Invalidate cache entries.

**Query params:**
- `?type=routing` - Clear routing cache only
- `?type=response` - Clear response cache only
- `?all=true` - Clear all caches (default)

**Response:**
```json
{
  "message": "Cache invalidated",
  "cleared": {
    "routing_entries": 45,
    "response_entries": 23
  }
}
```

### `GET /admin/explain`

Explain routing decision for a given prompt.

**Query params:**
- `?prompt=Your prompt here` (required)
- `?category=...` - Override category detection

**Response:**
```json
{
  "prompt": "Write a Python function for binary search",
  "detected_category": "coding",
  "complexity": 0.42,
  "selected_model": "codellama:34b",
  "scores": {
    "capability_score": 0.87,
    "benchmark_score": 0.81,
    "speed_score": 0.65,
    "final_score": 0.82
  },
  "alternatives_considered": [
    {"model": "llama3:70b", "score": 0.76, "rejected_reason": "Too slow"},
    {"model": "phi3:mini", "score": 0.45, "rejected_reason": "Below minimum size"}
  ]
}
```

---

## Error Codes

| HTTP Status | Meaning | Common Causes |
|-------------|---------|---------------|
| 200 | Success | - |
| 400 | Bad Request | Invalid JSON, missing required fields |
| 401 | Unauthorized | Invalid/missing admin API key |
| 404 | Not Found | Endpoint doesn't exist |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Backend failure, model load error |
| 503 | Service Unavailable | Backend not connected, all models failed |

## Rate Limits

- **General endpoints:** 60 requests/minute per IP (configurable)
- **Admin endpoints:** 10 requests/minute per IP (configurable)
- **Chat completions:** Also limited by backend provider rate limits

Rate limit exceeded responses include `Retry-After` header with seconds to wait.

## CORS

CORS is disabled by default. To enable for specific origins, set `ROUTER_CORS_ALLOWED_ORIGINS` in `.env`:

```
ROUTER_CORS_ALLOWED_ORIGINS=http://localhost:3000,https://myapp.example.com
```

---

## Client Compatibility

SmarterRouter is compatible with any OpenAI client library that supports:
- `/v1/chat/completions` endpoint
- `/v1/models` endpoint
- Optional: `/v1/embeddings` endpoint

**Tested with:**
- OpenAI Python SDK
- OpenWebUI (v0.2+)
- Continue (VS Code extension)
- Cursor IDE
- SillyTavern
- custom applications

**Important:** The router presents itself as a single model (`smarterrouter/main`) to simplify frontend integration. The actual model selection is transparent to the client.
