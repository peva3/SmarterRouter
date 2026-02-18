# SmarterRouter

An intelligent, multi-backend AI router that sits between your application and various LLM providers. It profiles your models, aggregates benchmark data, and intelligently routes each query to the best available model for the task.

## Features

- **Multi-Backend Support**: Works with Ollama, llama.cpp servers, and any OpenAI-compatible API.
- **Smart Routing**: Category-first routing (coding, reasoning, creativity) and complexity-aware model selection.
- **Smart Caching**: Thread-safe semantic similarity caching with embeddings + response caching (500 routing + 200 response entries) for identical prompts.
- **Category-Minimum Size**: Prevents small models from being selected for complex tasks (e.g., 0.5B for hard coding).
- **Vector Embeddings**: Full `/v1/embeddings` endpoint for RAG and semantic search applications.
- **Multimodal Support**: Automatically routes vision tasks to vision-capable models.
- **Tool Use Detection**: Routes function-calling requests to optimized models.
- **LLM-as-Judge Scoring**: Qualitative model evaluation using high-end models as judges (e.g., GPT-4o).
- **Standardized Benchmarks**: Uses MT-Bench style prompts for rigorous performance measurement.
- **VRAM Monitoring & Management**: Monitors GPU usage via nvidia-smi, measures per-model VRAM during profiling, and proactively unloads models to avoid OOM. Supports pinned model for low-latency responses. Includes `/admin/vram` endpoint for real-time monitoring.
- **Runtime Profiling**: Tests your actual models with real prompts to measure their capabilities on your hardware.
- **Security Hardening**:
  - **SQL Injection Prevention**: All database operations use ORM with whitelist validation.
  - **API Key Authentication**: Protects admin endpoints.
  - **Rate Limiting**: Thread-safe request throttling to prevent abuse.
  - **Input Validation & Sanitization**: Enforces schema validation, length limits, and sanitizes prompts.
  - **Log Sanitization**: Automatic redaction of secrets and API keys.
- **Frontend Integration**: Presents itself as a single, configurable model name for seamless integration with UIs like OpenWebUI.
- **Production Ready**: Comprehensive test suite (73 tests), thread-safe architecture, and extensive error handling.


## Quick Start

### Prerequisites
- Python 3.11+
- An LLM backend (Ollama, llama.cpp server, etc.)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd llm-router
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure:**
    Copy `ENV_DEFAULT` to `.env` and customize it for your setup.
    ```bash
    cp ENV_DEFAULT .env
    nano .env
    ```
4.  **Start the server:**
    ```bash
    python -m uvicorn main:app --host 0.0.0.0 --port 11436
    ```

### Docker
```bash
docker-compose up -d --build
```

## How It Works

### First Run (One-Time Setup)
When you start the router for the first time:
1.  **Discovery**: Fetches all models from your configured backend.
2.  **Profiling**: Tests each new model with prompts across multiple categories (reasoning, coding, etc.) to measure its performance on your hardware.
3.  **Benchmark Sync**: Downloads benchmark scores from HuggingFace and LMSYS. This process can take a while but only happens once per model.

### Normal Operation
1.  **Polling**: The router periodically checks for new models in your backend.
2.  **Routing**: When a request comes in:
    a.  The prompt is analyzed for its category (e.g., coding, reasoning) and complexity (difficulty prediction).
    b.  The router queries its database for benchmark data, runtime profiles, and **user feedback**.
    c.  A scoring algorithm selects the best model, balancing **quality vs. speed** based on your preference.
    d.  **Cascading/Fallback**: If the selected model fails, the system automatically retries with the next best capable model.
3.  **Dispatch**: The request is forwarded to the chosen model.
4.  **Response**: The response is returned to the user, with an optional signature identifying the model used.

## Configuration

All configuration is managed via environment variables (or the `.env` file). See `ENV_DEFAULT` for a full list of options.

**Key Settings:**
- `ROUTER_PROVIDER`: `ollama`, `llama.cpp`, or `openai`.
- `ROUTER_QUALITY_PREFERENCE`: 0.0 (Max Speed) to 1.0 (Max Quality). Default 0.5.
- `ROUTER_CASCADING_ENABLED`: Enable automatic fallback to other models on failure (default: true).
- `ROUTER_FEEDBACK_ENABLED`: Enable feedback collection to improve routing (default: true).
- `ROUTER_OLLAMA_URL`: The URL for your chosen backend.

## API Usage

### `/v1/feedback` (New)
Submit user feedback to improve future routing.
```bash
curl -X POST http://localhost:11436/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "chatcmpl-...",
    "score": 1.0,
    "comment": "Great answer!"
  }'
```

### `/v1/models`
Returns the router itself as a single model, which simplifies frontend integration.
```bash
curl http://localhost:11436/v1/models
```
**Example Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "smarterrouter/main",
      "object": "model",
      "created": 1708162374.0,
      "owned_by": "local",
      "description": "An intelligent router...",
      "admin_auth_required": true
    }
  ]
}
```

### `/v1/chat/completions`
Submit a prompt for routing and generation.
```bash
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function for a binary search tree."}]
  }'
```

### `/v1/embeddings` (New)
Generate vector embeddings for RAG and semantic search.
```bash
curl -X POST http://localhost:11436/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

### Admin Endpoints
- `/admin/profiles`: View the performance profiles of your models.
- `/admin/benchmarks`: View the aggregated benchmark data.
- `/admin/reprofile`: Trigger a manual reprofiling of all models.
- `/admin/vram`: View real-time VRAM usage, history, and model memory allocation.

**Example (with authentication):**
```bash
curl http://localhost:11436/admin/profiles \
  -H "Authorization: Bearer your-secret-key"
```

## Security
- **API Key Authentication**: Protects admin endpoints.
- **Rate Limiting**: Prevents abuse with configurable request limits.
- **Input Validation**: All requests are validated against Pydantic schemas, with strict length limits and sanitization.
- **SQL Injection Prevention**: All database queries use the SQLAlchemy ORM.
- **Log Sanitization**: Sensitive data like API keys are redacted from logs.

## Supported Backends

| Backend | Use Case | Features |
|---|---|---|
| Ollama | Local inference | Full support, including VRAM management. |
| llama.cpp | Self-hosted `llama.cpp` | OpenAI-compatible endpoints. |
| OpenAI | External APIs | API key authentication. |

## License

MIT License
