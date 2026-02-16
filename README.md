# LLM Router Proxy

An intelligent, multi-backend AI router that sits between your application and various LLM providers. It profiles your models, aggregates benchmark data, and intelligently routes each query to the best available model for the task.

## Features

- **Multi-Backend Support**: Works with Ollama, llama.cpp servers, and any OpenAI-compatible API.
- **Smart Routing**: Category-first routing (coding, reasoning, creativity) and complexity-aware model selection.
- **Benchmark Integration**: Pulls data from HuggingFace Leaderboard and LMSYS Chatbot Arena to make data-driven decisions.
- **VRAM Management**: Proactive model unloading for systems with limited GPU memory, with a "pinned model" option for speed.
- **Runtime Profiling**: Tests your actual models with real prompts to measure their capabilities on your hardware.
- **Security Hardening**:
  - **API Key Authentication**: Protects admin endpoints.
  - **Rate Limiting**: Throttles requests to prevent abuse.
  - **Input Validation & Sanitization**: Enforces schema validation, length limits, and sanitizes prompts.
- **Frontend Integration**: Presents itself as a single, configurable model name for seamless integration with UIs like OpenWebUI.

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
    a.  The prompt is analyzed for its category (e.g., coding, reasoning) and complexity.
    b.  The router queries its database for benchmark data and runtime profiles of available models.
    c.  A scoring algorithm selects the best model, prioritizing models with strong benchmark data for the prompt's category and matching model size to prompt complexity.
3.  **Dispatch**: The request is forwarded to the chosen model.
4.  **Response**: The response is returned to the user, with an optional signature identifying the model used.

## Configuration

All configuration is managed via environment variables (or the `.env` file). See `ENV_DEFAULT` for a full list of options.

**Key Settings:**
- `ROUTER_PROVIDER`: `ollama`, `llama.cpp`, or `openai`.
- `ROUTER_OLLAMA_URL`, `ROUTER_LLAMA_CPP_URL`, `ROUTER_OPENAI_BASE_URL`: The URL for your chosen backend.
- `ROUTER_EXTERNAL_MODEL_NAME`: The name the router presents to frontends (e.g., `hubrouter/main`).
- `ROUTER_ADMIN_API_KEY`: Set this to a secure key to protect admin endpoints.
- `ROUTER_GENERATION_TIMEOUT`: How long to wait for a model to generate a response (default: 120s).

## API Usage

The router exposes an OpenAI-compatible API.

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
      "id": "hubrouter/main",
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

### Admin Endpoints
- `/admin/profiles`: View the performance profiles of your models.
- `/admin/benchmarks`: View the aggregated benchmark data.
- `/admin/reprofile`: Trigger a manual reprofiling of all models.

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
