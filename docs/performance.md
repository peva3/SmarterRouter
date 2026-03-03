# Performance Tuning Guide

Optimize SmarterRouter for your specific workload and hardware.

## Table of Contents
- [Latency vs Throughput](#latency-vs-throughput)
- [Configuration Trade-offs](#configuration-trade-offs)
- [Profiling Optimization](#profiling-optimization)
- [VRAM Management](#vram-management)
- [Database Performance](#database-performance)
- [Monitoring](#monitoring)

## Latency vs Throughput

### For Low Latency (< 500ms P99)

**Goal:** Fastest possible response times for interactive use.

**Settings:**
```env
# Pin a small model (1-3B) for instant responses
ROUTER_PINNED_MODEL=phi3:mini

# Prefer smaller models
ROUTER_QUALITY_PREFERENCE=0.2
ROUTER_PREFER_SMALLER_MODELS=true

# Enable response caching
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_MAX_SIZE=1000
ROUTER_CACHE_TTL_SECONDS=3600

# Fast profiling (less accurate)
ROUTER_PROFILE_PROMPTS_PER_CATEGORY=1

# Reduce cache complexity (exact match only)
ROUTER_EMBED_MODEL=  # leave empty
```

**Hardware:** GPU with at least 8GB VRAM for pinned model + one additional model.

**Expected results:**
- First token: ~200-500ms (if model already loaded)
- Full response (100 tokens): ~500ms - 2s

---

### For High Throughput (> 100 RPM)

**Goal:** Maximize requests per minute, minimize cost.

**Settings:**
```env
# Use batch processing where possible
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_MAX_SIZE=2000
ROUTER_CACHE_RESPONSE_MAX_SIZE=500

# Enable semantic caching if you have embedding model
ROUTER_EMBED_MODEL=nomic-embed-text:latest
ROUTER_CACHE_SIMILARITY_THRESHOLD=0.80

# Aggressive caching
ROUTER_CACHE_TTL_SECONDS=7200  # 2 hours

# Rate limiting to prevent overload
ROUTER_RATE_LIMIT_ENABLED=true
ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=200

# Reduce logging overhead
ROUTER_LOG_FORMAT=json
ROUTER_LOG_LEVEL=WARNING
```

**Additional:**
- Use load balancer in front of multiple router instances
- Use PostgreSQL instead of SQLite for concurrent access
- Increase worker processes if using gunicorn

**Expected results:**
- 100+ requests/minute with cached responses
- 30-50 requests/minute with fresh generations

---

### For High Quality

**Goal:** Always select the best possible model, regardless of speed.

**Settings:**
```env
ROUTER_QUALITY_PREFERENCE=1.0
ROUTER_JUDGE_ENABLED=true
ROUTER_JUDGE_MODEL=gpt-4o
ROUTER_PROFILE_PROMPTS_PER_CATEGORY=5  # more profiling samples
ROUTER_CACHE_ENABLED=false  # don't compromise quality for speed
```

**Trade-offs:** Slower responses, higher VRAM usage.

---

## Configuration Trade-offs

| Setting | Speed Impact | Quality Impact | Memory Impact |
|---------|--------------|----------------|---------------|
| `ROUTER_QUALITY_PREFERENCE=0.0` | ✅✅✅ | ↓↓↓ | - |
| `ROUTER_QUALITY_PREFERENCE=1.0` | ↓↓↓ | ✅✅✅ | - |
| `ROUTER_CACHE_ENABLED=true` | ✅✅✅ | ↔️ | ↔️ |
| `ROUTER_PINNED_MODEL=phi3:mini` | ✅✅ | ↓ | ↑↑↑ |
| `ROUTER_PROFILE_PROMPTS_PER_CATEGORY=1` | ✅ | ↓ | - |
| `ROUTER_JUDGE_ENABLED=true` | - | ✅✅ | - |

---

## Profiling Optimization

Profiling is one-time cost but can be significant with many models.

### Reduce Profiling Time

```env
# Fewer prompts per category (default=3)
ROUTER_PROFILE_PROMPTS_PER_CATEGORY=1

# Skip LLM-as-Judge (much faster, less accurate)
ROUTER_JUDGE_ENABLED=false

# Increase timeout to avoid retries
ROUTER_PROFILE_TIMEOUT=180
```

**Expected speedup:**
- Default (3 prompts + judge): ~30-60 min for 18 models
- Optimized (1 prompt, no judge): ~10-20 min for 18 models

### Parallel Profiling

SmarterRouter profiles models in parallel by default (limited by `ROUTER_PROFILE_CONCURRENCY` if needed). Control concurrency:

```env
# Maximum concurrent profiling operations
ROUTER_PROFILE_CONCURRENCY=4
```

Higher values = faster profiling but more VRAM/CPU load.

---

## VRAM Management

### Optimize VRAM Usage

```env
# Set realistic limit (leave 10-15% headroom)
ROUTER_VRAM_MAX_TOTAL_GB=22.0  # for 24GB GPU

# Aggressive unloading
ROUTER_VRAM_AUTO_UNLOAD_ENABLED=true
ROUTER_VRAM_UNLOAD_THRESHOLD_PCT=75
ROUTER_VRAM_UNLOAD_STRATEGY=largest

# Don't pin too many models
# Only ROUTER_PINNED_MODEL is truly pinned; others unload automatically
```

### Monitor VRAM

```bash
curl http://localhost:11436/admin/vram | jq .
```

Watch for:
- `utilization_pct > 90%` → risk of OOM
- `warnings` array → threshold exceeded
- `free_gb < 2` → load smaller models or increase limit

### Reduce VRAM Footprint

1. **Use quantized models:** `Q4_K_M`, `Q5_K_S` variants
2. **Prefer smaller base models:** `1B`, `3B`, `7B` over `70B`
3. **Adjust context size:** `OLLAMA_NUM_CTX=2048` reduces memory
4. **Unload unused models manually:**
   ```bash
   # Restart with only necessary models, or adjust ROUTER_PINNED_MODEL
   ```

---

## Database Performance

### SQLite Tuning

For write-heavy workloads with many models:

```env
# Increase cache size
ROUTER_DATABASE_URL=sqlite:///data/router.db?cache=shared&cache_size=10000

# Use WAL mode (better concurrency)
# Add to startup script or custom code:
#   engine._engine.execute('PRAGMA journal_mode=WAL')
```

### PostgreSQL (Production)

```env
ROUTER_DATABASE_URL=postgresql://user:pass@localhost:5432/smarterrouter
```

Benefits:
- Better concurrency
- Faster writes
- Easier backups
- Connection pooling

**Setup:**
```bash
docker run -d \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=smarterrouter \
  -p 5432:5432 \
  postgres:15
```

---

## Monitoring

### Prometheus Metrics

Scrape `http://localhost:11436/metrics` every 30 seconds.

**Key metrics to alert on:**
- `smarterrouter_request_duration_seconds{endpoint="/v1/chat/completions"}` - P95 > 10s?
- `smarterrouter_errors_total` - Rising error rate
- `smarterrouter_vram_utilization_pct` - > 85%?
- `smarterrouter_cache_hit_rate` - < 50% suggests cache size too small

### Log Aggregation

Use `ROUTER_LOG_FORMAT=json` and ship to:
- Loki/Grafana
- ELK stack
- Datadog
- CloudWatch

**Example log entry:**
```json
{
  "timestamp": "2024-02-20T12:34:56.789Z",
  "level": "INFO",
  "request_id": "abc123",
  "endpoint": "/v1/chat/completions",
  "model_selected": "llama3:70b",
  "latency_ms": 1250,
  "cache_hit": false,
  "vram_used_gb": 18.2
}
```

---

## Benchmark Your Setup

To measure improvements:

```bash
# Install hey (HTTP load testing)
go install github.com/rakyll/hey@latest

# Test with 100 requests, 10 concurrent
hey -c 10 -n 100 \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:11436/v1/chat/completions
```

Track:
- **Avg latency:** Target < 2s (first token < 500ms)
- **RPS:** Requests per second
- **Error rate:** Should be 0%

Compare before/after config changes to validate tuning.
