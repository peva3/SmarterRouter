# LLM Router Proxy

An intelligent AI router that sits between OpenWebUI and various LLM backends. It profiles your models, aggregates benchmark data from multiple sources, and intelligently routes each query to the best available model.

## Features

- **Multi-Backend Support**: Works with Ollama, llama.cpp server, and any OpenAI-compatible API
- **Smart Routing**: Category-first routing (coding, reasoning, creativity, factual)
- **Benchmark Integration**: Pulls data from HuggingFace Leaderboard and LMSYS Chatbot Arena
- **VRAM Management**: Proactive model unloading for limited GPU memory
- **Runtime Profiling**: Tests actual models to measure capabilities
- **Response Signatures**: Transparent model identification on every response

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (see Configuration section)
# Edit .env with your backend settings

# Start the router
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436
```

### Docker

```bash
docker-compose up -d --build
```

## Configuration

```bash
# Provider selection (ollama, llama.cpp, openai)
ROUTER_PROVIDER=ollama

# === Ollama settings (default) ===
ROUTER_OLLAMA_URL=http://localhost:11434
ROUTER_PINNED_MODEL=phi3:mini  # Keep small model in VRAM

# === llama.cpp server settings ===
# ROUTER_PROVIDER=llama.cpp
# ROUTER_LLAMA_CPP_URL=http://localhost:8080

# === OpenAI-compatible API settings ===
# ROUTER_PROVIDER=openai
# ROUTER_OPENAI_BASE_URL=https://api.openai.com/v1
# ROUTER_OPENAI_API_KEY=your-api-key
# ROUTER_MODEL_PREFIX=

# Server settings
ROUTER_HOST=0.0.0.0
ROUTER_PORT=11434

# Security settings (optional but recommended)
ROUTER_ADMIN_API_KEY=your-secret-key-here  # Protect admin endpoints
ROUTER_RATE_LIMIT_ENABLED=false  # Enable rate limiting
ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=60  # General endpoint limit
ROUTER_RATE_LIMIT_ADMIN_REQUESTS_PER_MINUTE=10  # Admin endpoint limit

# Generation timeout (seconds) - larger models need more time
ROUTER_GENERATION_TIMEOUT=120

# Routing mode
ROUTER_MODEL=  # Leave empty for keyword-based, or set to a model for LLM dispatch

# Benchmark sources
ROUTER_BENCHMARK_SOURCES=huggingface,lmsys
```

## Supported Backends

| Backend | Use Case | Features |
|---------|----------|----------|
| Ollama | Local inference | Full support, VRAM management |
| llama.cpp | llama.cpp server, llama-swap | OpenAI-compatible endpoints |
| OpenAI | External APIs | API key auth, any compatible server |

## API Usage

```bash
# Health check
curl http://localhost:11434/health

# List models
curl http://localhost:11434/v1/models

# Chat completion
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

### Admin Endpoints

Admin endpoints can be protected with an API key for security:

```bash
# Without API key (if ROUTER_ADMIN_API_KEY not set)
curl http://localhost:11434/admin/profiles

# With API key
curl http://localhost:11434/admin/profiles \
  -H "Authorization: Bearer your-secret-key"

# View benchmarks
curl http://localhost:11434/admin/benchmarks \
  -H "Authorization: Bearer your-secret-key"

# Trigger reprofile
curl -X POST http://localhost:11434/admin/reprofile \
  -H "Authorization: Bearer your-secret-key"
```

### Security Features

- **API Key Authentication**: Protect admin endpoints with `ROUTER_ADMIN_API_KEY`
- **Rate Limiting**: Optional request throttling per client IP
  - Enable with `ROUTER_RATE_LIMIT_ENABLED=true`
  - Configure limits with `ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE` and `ROUTER_RATE_LIMIT_ADMIN_REQUESTS_PER_MINUTE`
- **Input Validation**: Pydantic models validate all incoming requests
  - Content-Type validation (must be `application/json`)
  - Request body schema validation
  - Length limits (prompts max 10,000 characters, max 100 messages)
  - Role validation (only `user`, `assistant`, `system` allowed)
- **SQL Injection Prevention**: All database queries use ORM with parameterized queries
- **Prompt Sanitization**: Automatic removal of null bytes and control characters
- **Log Sanitization**: API keys and sensitive data are redacted from logs
- **Backward Compatible**: If no API key is set, admin endpoints remain open (existing behavior)

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Architecture

```
User Prompt → Router → Backend (Ollama/llama.cpp/OpenAI) → Response + Signature
                ↓
         Benchmark Data (HuggingFace, LMSYS)
                ↓
         Runtime Profiles
```

## Tech Stack

- Python 3.11+
- FastAPI
- SQLAlchemy + SQLite
- httpx (async HTTP)

## License

MIT License
