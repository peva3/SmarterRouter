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
    cd SmarterRouter
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

The easiest way to run SmarterRouter is with Docker Compose.

**Prerequisites:**
- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with drivers and NVIDIA Container Toolkit for VRAM monitoring

**1. Create a `.env` file**

Copy the environment template and customize:

```bash
cp ENV_DEFAULT .env
# Edit .env to set your backend URL, API keys, etc.
```

At minimum, set `ROUTER_OLLAMA_URL` if your Ollama is not on `localhost:11434`.

**2. Run with Docker Compose**

⚠️ **Important:** You must create a `.env` file first (see step 1).

```bash
docker-compose up -d
```

This will pull the latest image from GitHub Container Registry and start SmarterRouter on `http://localhost:11436`.

**3. (Optional) Enable GPU Support**

The `docker-compose.yml` includes GPU configuration in the `deploy` section. However, this is only activated when using the `--compatibility` flag (or in Docker Swarm mode). There are two ways to enable GPU:

```bash
# Method A: Use --compatibility (respects the deploy section)
docker-compose --compatibility up -d

# Method B: Use --gpus flag (newer Docker Compose)
docker compose up -d --gpus all
```

Both achieve the same result. If you don't need GPU/VRAM monitoring, skip this step.

**4. Verify it's running**

```bash
docker logs smarterrouter
# Look for: "Uvicorn running on http://0.0.0.0:11436"
```

**docker-compose.yml reference:**

```yaml
version: "3.8"

services:
  smarterrouter:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: smarterrouter
    ports:
      - "11436:11436"
    env_file:
      - .env
    volumes:
      - ./router.db:/app/router.db
    restart: unless-stopped
    networks:
      - smarterrouter-network
    # Uncomment for GPU support (requires NVIDIA Container Toolkit)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

networks:
  smarterrouter-network:
    driver: bridge
```

**docker run** (no compose):

```bash
docker run -d \
  --name smarterrouter \
  -p 11436:11436 \
  --env-file .env \
  -v $(pwd)/router.db:/app/router.db \
  smarterrouter:latest
```

Add `--gpus all` for GPU support.

**Note:** The Dockerfile includes labels pointing to the upstream repository. If you fork and publish your own images, consider updating the `org.opencontainers.image.url` and `org.opencontainers.image.source` labels to point to your fork.

---

### GPU Support Details

SmarterRouter uses `nvidia-smi` inside the container to monitor VRAM. This requires:

- NVIDIA GPU with proprietary drivers installed on the host
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html>
- Verify installation: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`

If `nvidia-smi` is not available, the VRAM monitor will be disabled and you'll see a warning in the logs. The router will still function but won't measure or manage VRAM.

---

### Configuration

All configuration is via the `.env` file (see `ENV_DEFAULT` for full list). Important settings:

- `ROUTER_OLLAMA_URL`: URL of your Ollama/backend (default: `http://localhost:11434`)
- `ROUTER_PROVIDER`: `ollama`, `llama.cpp`, or `openai`
- `ROUTER_VRAM_MAX_TOTAL_GB`: Set to limit total VRAM usage (auto-detects 90% of GPU if unset)
- `ROUTER_PINNED_MODEL`: Keep a specific model always loaded for fast responses

The database (`router.db`) is persisted in your current directory via the volume mount. This means your model profiles, routing history, and learned feedback survive container upgrades and recreations. **Back up this file regularly** to preserve your router's state.

---

### Troubleshooting

**"Failed to list models: All connection attempts failed"**
- SmarterRouter cannot reach your backend (Ollama/llama.cpp). Check `ROUTER_OLLAMA_URL` and ensure the backend is running and accessible from the container.

**Port already in use**
- Change the host port mapping (`"11436:11436"`) or stop the existing container.

**GPU not working / nvidia-smi not found**
- Install NVIDIA Container Toolkit and restart Docker daemon.
- Use `--gpus all` flag or enable `deploy.resources` in compose.
- Test with: `docker exec smarterrouter nvidia-smi`

**Permissions errors on router.db**
- Ensure the host directory is writable by the Docker user (UID 1000). The container runs as root by default.

**Container exits immediately**
- Check logs: `docker logs smarterrouter`
- Common causes: invalid `.env` syntax, missing required files, port conflict.

**Need to reset the database**
- Stop the container and delete `router.db` from your current directory. It will be recreated on next startup. Warning: You will lose all model profiles and history; you'll need to re-profile your models.

---

### Updating

Pull the latest image and recreate:

```bash
docker pull ghcr.io/<YOUR_USERNAME>/smarterrouter:latest
docker-compose down
docker-compose up -d
```

**Note:** Your `router.db` file is preserved and will be reused automatically. If the new version includes database schema changes, the existing database will be automatically migrated on startup (if needed). No manual intervention required in most cases.

If you encounter database-related errors after an upgrade, you can delete `router.db` to start fresh (you will lose all profiles and history, requiring a full re-profile).

---

### OpenWebUI Integration

OpenWebUI works seamlessly with SmarterRouter as a drop-in OpenAI-compatible backend.

1. **Start SmarterRouter** using one of the methods above.
2. **Open OpenWebUI** → Admin Settings → **Connections** → **Add a Connection**.
3. **Configure the connection**:
   - **Name**: `SmarterRouter`
   - **Base URL**: `http://localhost:11436/v1` (adjust host/port if needed)
   - **API Key**: Leave empty (SmarterRouter does not require a key for chat)
4. **Save** and select the model `smarterrouter/main` from the model dropdown, it should be enabled by default in the admin settings models page.

OpenWebUI will now route all conversations through SmarterRouter, which automatically selects the best model for each prompt from your installed Ollama/backend models.

---

## How It Works

### First Run (One-Time Setup)
When you start the router for the first time:

1.  **Discovery**: Fetches all models from your configured backend.
2.  **Profiling**: Tests each model with prompts across multiple categories (reasoning, coding, etc.) to measure its performance on your hardware. This is a one-time process. For ~18 models, expect about 1.5 hours of profiling time. Subsequent runs only profile *new* models added to your backend.
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
