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

| Vendor | Detection Method | Docker Support | Template |
|--------|-----------------|----------------|----------|
| NVIDIA | nvidia-smi | ✅ Full | [docker-compose.nvidia.yml](docker-compose.nvidia.yml) |
| AMD | rocm-smi or sysfs | ✅ ROCm containers | [docker-compose.amd.yml](docker-compose.amd.yml) |
| Intel Arc | sysfs (lmem) | ⚠️ Limited | [docker-compose.intel.yml](docker-compose.intel.yml) |
| Apple Silicon | Unified memory | ❌ Run on host | [docker-compose.apple.md](docker-compose.apple.md) |
| Multi-GPU | Combined detection | ✅ Mixed vendors | [docker-compose.multi-gpu.yml](docker-compose.multi-gpu.yml) |

**Quick Start:** Copy the appropriate template to your project root:
```bash
# For NVIDIA GPUs
cp docs/docker-compose.nvidia.yml docker-compose.yml

# For AMD GPUs  
cp docs/docker-compose.amd.yml docker-compose.yml

# For Intel Arc GPUs
cp docs/docker-compose.intel.yml docker-compose.yml

# For multi-GPU setups
cp docs/docker-compose.multi-gpu.yml docker-compose.yml
```

### NVIDIA GPUs (Recommended)

NVIDIA provides the best Docker GPU support with the NVIDIA Container Toolkit.

**Requirements:**
- NVIDIA GPU with proprietary drivers installed
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html>

**Quick Setup:**
```bash
# Use the NVIDIA template
cp docs/docker-compose.nvidia.yml docker-compose.yml
docker-compose --compatibility up -d
```

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

**Quick Setup:**
```bash
# Use the AMD template
cp docs/docker-compose.amd.yml docker-compose.yml
docker-compose up -d
```

**Verify installation:**
```bash
# Check if rocm-smi is available
rocm-smi

# Or check sysfs
ls /sys/class/drm/card*/device/mem_info_vram_total
```
```

**Note:** For full ROCm support in containers, you may need to use a ROCm base image. See [docker-compose.amd.yml](docker-compose.amd.yml) for detailed options.

#### AMD APUs (Unified Memory)

AMD APUs (Accelerated Processing Units) like Ryzen AI 300 series with Radeon 800M graphics use **unified memory** where CPU and GPU share system RAM. This requires special configuration.

**Supported APUs:**
- Ryzen AI 9 HX 370 (Radeon 890M)
- Ryzen AI 9 HX 470 (Radeon 890M)
- Ryzen 8000G series (Radeon 780M/760M)
- Ryzen 5000/6000 mobile series with Radeon Graphics

**Auto-Detection:**
SmarterRouter automatically detects APUs and uses GTT (Graphics Translation Table) to report the unified memory pool, not the small BIOS VRAM carve-out.

**BIOS Configuration (Critical for APUs):**
1. Enter BIOS/UEFI settings
2. Find "UMA Frame Buffer Size" or "UMA Mode" (often under Advanced > NB Configuration)
3. **Set to minimum (512MB - 2GB)** - NOT maximum!
   - Why? The BIOS setting is a VRAM *carve-out* that reduces available system RAM
   - APUs use GTT for actual GPU memory, which dynamically allocates from system RAM
   - Large carve-out just wastes RAM; GTT pool is the real usable memory
4. Save and reboot

**Manual Override (if auto-detection fails):**
```bash
# In .env - set to ~90% of your system RAM for the GPU
# Example: 64GB system -> set ~58GB
ROUTER_AMD_UNIFIED_MEMORY_GB=58
```

**Verification:**
```bash
# Check GTT pool size (actual unified memory)
cat /sys/class/drm/card*/device/mem_info_gtt_total
# Divide by 1073741824 to get GB

# Check VRAM carve-out (usually small for APUs)
cat /sys/class/drm/card*/device/mem_info_vram_total
```

**Architecture Override (for gfx1150/gfx1151):**
Some newer APUs need a ROCm architecture override:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

### Intel Arc GPUs

Intel Arc GPUs use sysfs for memory monitoring via local memory (lmem).

**Requirements:**
- Intel Arc A-series GPU (A380, A770, etc.) or Data Center GPU
- Intel GPU drivers (i915 kernel module)

**Quick Setup:**
```bash
# Use the Intel template
cp docs/docker-compose.intel.yml docker-compose.yml
docker-compose up -d
```

**Verify installation:**
```bash
# Check for Intel GPU with dedicated memory
ls /sys/class/drm/card*/device/lmem_total
```

**Note:** Intel GPU support in Docker requires the device to be passed through. Compute workloads may need oneAPI/Level Zero setup. See [docker-compose.intel.yml](docker-compose.intel.yml) for details.

### Apple Silicon (M1/M2/M3)

Apple Silicon uses unified memory where CPU and GPU share system RAM. VRAM monitoring estimates GPU availability as 75% of total RAM.

**Important:** Docker Desktop on macOS **cannot pass GPU to containers**. You must run SmarterRouter directly on the host (not in Docker) for Apple Silicon GPU support.

See [docker-compose.apple.md](docker-compose.apple.md) for complete native installation instructions.

**Quick Start:**
```bash
# Clone and setup
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure for Apple Silicon (optional)
echo "ROUTER_APPLE_UNIFIED_MEMORY_GB=16" >> .env  # Set if auto-detect fails

# Run
python -m uvicorn main:app --host 0.0.0.0 --port 11436
```

### Multi-GPU Setups

SmarterRouter automatically detects all GPUs across vendors. For mixed GPU setups (e.g., NVIDIA + AMD, or all three), use the multi-GPU template:

```bash
# Copy the multi-GPU template
cp docs/docker-compose.multi-gpu.yml docker-compose.yml

# Edit to uncomment the sections for your specific GPUs
nano docker-compose.yml

# Run with NVIDIA support (if included)
docker-compose --compatibility up -d
```

**Multi-GPU configuration:**
- Set `ROUTER_VRAM_MAX_TOTAL_GB` to limit total VRAM usage across all GPUs
- GPUs are indexed globally (0, 1, 2, ...) regardless of vendor
- Check `/admin/vram` endpoint to see detected GPUs

**Example combinations:**
- **NVIDIA + AMD**: Uncomment both deploy.resources (NVIDIA) and devices (AMD)
- **NVIDIA + Intel**: Uncomment deploy.resources (NVIDIA) and devices (Intel)
- **AMD + Intel**: Uncomment devices section only
- **All three**: Uncomment all GPU sections

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
