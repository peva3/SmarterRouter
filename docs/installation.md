# Installation Guide

This guide covers installing and running SmarterRouter in various environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Docker (Recommended)](#docker-recommended)
- [Manual Installation](#manual-installation)
- [GPU Support](#gpu-support)
- [Verification](#verification)
- [Updating](#updating)

## Prerequisites

### For Docker
- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with drivers and NVIDIA Container Toolkit for VRAM monitoring

### For Manual Installation
- Python 3.11+
- pip package manager
- Access to an LLM backend (Ollama, llama.cpp server, or OpenAI-compatible API)

## Docker (Recommended)

The easiest and most reliable way to run SmarterRouter.

### 1. Clone Repository

```bash
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter
```

### 2. Configure Environment

Copy the environment template and customize:

```bash
cp ENV_DEFAULT .env
nano .env  # or use your preferred editor
```

**Minimum configuration:**
- If your Ollama runs on `localhost:11434` (default), no changes needed
- If Ollama is on a different host/port, set `ROUTER_OLLAMA_URL`

**Important Docker networking note:** When running SmarterRouter in Docker and Ollama on the host machine, use `http://172.17.0.1:11434` instead of `http://localhost:11434` because localhost inside the container refers to the container itself.

### 3. Start the Router

```bash
docker-compose up -d
```

This will:
- Build the Docker image from GitHub Container Registry
- Start SmarterRouter on `http://localhost:11436`
- Mount the database at `./router.db` for persistence

### 4. Verify Installation

```bash
docker logs smarterrouter
```

Look for:
```
INFO:     Uvicorn running on http://0.0.0.0:11436
INFO:     Starting router...
INFO:     Profiling complete - X models ready
```

## Manual Installation

### 1. Clone Repository

```bash
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp ENV_DEFAULT .env
nano .env
```

At minimum, verify `ROUTER_OLLAMA_URL` points to your backend.

### 5. Start the Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 11436
```

**For production:** Use a production ASGI server like gunicorn:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:11436
```

## GPU Support

SmarterRouter supports automatic VRAM monitoring across multiple GPU vendors:

| Vendor | Detection Method | Docker Support |
|--------|-----------------|----------------|
| NVIDIA | nvidia-smi | ✅ Full (NVIDIA Container Toolkit) |
| AMD | rocm-smi or sysfs | ✅ ROCm containers |
| Intel Arc | sysfs (lmem) | ⚠️ Limited (oneAPI/Level Zero) |
| Apple Silicon | Unified memory | ❌ Run on host (no GPU passthrough) |

### NVIDIA GPUs (Recommended)

NVIDIA provides the best Docker GPU support with the NVIDIA Container Toolkit.

**Requirements:**
- NVIDIA GPU with proprietary drivers installed
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html>

**Verify installation:**
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Enable GPU in Docker:**
```bash
# Method A: Use --compatibility flag
docker-compose --compatibility up -d

# Method B: Use --gpus flag (newer Docker Compose)
docker compose up -d --gpus all
```

### AMD GPUs (ROCm)

AMD GPUs use ROCm (Radeon Open Compute) for GPU monitoring.

**Requirements:**
- AMD GPU with ROCm support (RX 6000/7000 series, Radeon Instinct, Radeon Pro)
- ROCm runtime installed on host

**Verify installation:**
```bash
# Check if rocm-smi is available
rocm-smi

# Or check sysfs
ls /sys/class/drm/card*/device/mem_info_vram_total
```

**Docker configuration:**
Edit `docker-compose.yml` and uncomment the AMD GPU section, or use:
```bash
docker run -d \
  --name smarterrouter \
  --device /dev/kfd --device /dev/dri \
  -p 11436:11436 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ghcr.io/peva3/smarterrouter:latest
```

**Note:** For full ROCm support in containers, you may need to use a ROCm base image. See the `docker-compose.yml` for detailed options.

### Intel Arc GPUs

Intel Arc GPUs use sysfs for memory monitoring via local memory (lmem).

**Requirements:**
- Intel Arc A-series GPU (A380, A770, etc.) or Data Center GPU
- Intel GPU drivers (i915 kernel module)

**Verify installation:**
```bash
# Check for Intel GPU with dedicated memory
ls /sys/class/drm/card*/device/lmem_total
```

**Docker configuration:**
```bash
docker run -d \
  --name smarterrouter \
  --device /dev/dri \
  -p 11436:11436 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ghcr.io/peva3/smarterrouter:latest
```

**Note:** Intel GPU support in Docker requires the device to be passed through. Compute workloads may need oneAPI/Level Zero setup.

### Apple Silicon (M1/M2/M3)

Apple Silicon uses unified memory where CPU and GPU share system RAM. VRAM monitoring estimates GPU availability as 75% of total RAM.

**Important:** Docker Desktop on macOS **cannot pass GPU to containers**. You must run SmarterRouter directly on the host (not in Docker) for Apple Silicon GPU support.

**Native installation:**
```bash
# Clone and setup
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure for Apple Silicon
echo "ROUTER_APPLE_UNIFIED_MEMORY_GB=16" >> .env  # Optional: set if auto-detect fails

# Run
python -m uvicorn main:app --host 0.0.0.0 --port 11436
```

**Configuration options:**
- `ROUTER_APPLE_UNIFIED_MEMORY_GB`: Override auto-detected RAM (e.g., 16 for 16GB Mac)
- Default GPU allocation: 75% of system RAM

### Multi-GPU Setups

SmarterRouter automatically detects all GPUs across vendors:

```bash
# Mixed NVIDIA + AMD setup
# Both will be detected and combined for VRAM tracking
docker run -d \
  --name smarterrouter \
  --gpus all \
  --device /dev/kfd --device /dev/dri \
  -p 11436:11436 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ghcr.io/peva3/smarterrouter:latest
```

**Multi-GPU configuration:**
- Set `ROUTER_VRAM_MAX_TOTAL_GB` to limit total VRAM usage
- GPUs are indexed globally (0, 1, 2, ...) regardless of vendor
- Check `/admin/vram` endpoint to see detected GPUs

### No GPU / CPU-Only Mode

If no GPU is detected, SmarterRouter continues to function but:
- No VRAM monitoring available
- No automatic model unloading based on memory
- All model management falls back to the backend (Ollama, etc.)

To explicitly disable VRAM monitoring:
```env
ROUTER_VRAM_MONITOR_ENABLED=false
```

### GPU Support Feature Matrix

| Feature | NVIDIA | AMD | Intel Arc | Apple Silicon |
|---------|--------|-----|-----------|---------------|
| VRAM Detection | ✅ | ✅ | ✅ | ✅ (estimated) |
| Memory Usage | ✅ | ✅ | ✅ | ⚠️ (estimated) |
| Docker GPU Passthrough | ✅ | ⚠️ | ⚠️ | ❌ |
| Multi-GPU | ✅ | ✅ | ✅ | N/A |
| Model Auto-Unload | ✅ | ✅ | ✅ | ✅ |
| Pinned Model | ✅ | ✅ | ✅ | ✅ |

## Verification

### Check Health Endpoint

```bash
curl http://localhost:11436/health
```

Expected response:
```json
{
  "status": "healthy",
  "profiling_complete": true,
  "models_available": 5,
  "backend_connected": true
}
```

### Check Models Endpoint

```bash
curl http://localhost:11436/v1/models
```

Should return:
```json
{
  "object": "list",
  "data": [
    {
      "id": "smarterrouter/main",
      "object": "model",
      "created": 1708162374.0,
      "owned_by": "local"
    }
  ]
}
```

### Test Basic Request

```bash
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

## First Run: Profiling

On first startup, SmarterRouter:

1. **Discovers** all models from your configured backend
2. **Profiles** each model with standardized prompts (reasoning, coding, creativity)
3. **Downloads** benchmark data from HuggingFace and LMSYS
4. **Initializes** the routing database

**Expected timeline:**
- ~18 models: 30-60 minutes
- ~50 models: 2-4 hours
- Profile progress is logged; check logs with `docker logs -f smarterrouter`

**Profiling is one-time only.** Subsequent startups only profile newly added models.

## Updating

### Docker

```bash
docker pull ghcr.io/peva3/smarterrouter:latest
docker-compose down
docker-compose up -d
```

Your `router.db` file is preserved automatically. Database migrations happen on startup if needed.

### Manual

```bash
git pull
pip install -r requirements.txt
# Restart your server
```

## Uninstallation

### Docker

```bash
docker-compose down
docker volume rm smarterrouter_router-db  # if you want to delete the database
docker rmi ghcr.io/peva3/smarterrouter:latest
```

### Manual

```bash
# Stop the process (Ctrl+C or kill)
rm -rf venv router.db data/
```

## Next Steps

- [Configuration Reference](configuration.md) - All available settings
- [Backend Providers](backends.md) - Setting up different backends
- [API Documentation](api.md) - Complete API reference
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
