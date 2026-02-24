# Docker Setup Guide

This guide covers deploying SmarterRouter using Docker and Docker Compose, including both local-only and external provider setups.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Local Setup (Ollama)](#local-setup-ollama)
- [External Providers Only](#external-providers-only)
- [Mixed Local + External](#mixed-local--external)
- [Docker Commands](#docker-commands)
- [GPU Configuration](#gpu-configuration)
- [Production Considerations](#production-considerations)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter

# 2. Copy environment template
cp ENV_DEFAULT .env

# 3. Edit .env with your settings (optional)
nano .env

# 4. Start with Docker Compose
docker-compose up -d

# 5. Verify it's running
curl http://localhost:11436/health
```

That's it! SmarterRouter will auto-discover your Ollama models and start routing.

---

## Prerequisites

### Required
- Docker (20.10+)
- Docker Compose (v2.0+)

### Optional (for local models)
- **NVIDIA GPU**: NVIDIA drivers + NVIDIA Container Toolkit
- **AMD GPU**: ROCm runtime (RX 6000/7000 series+)
- **Intel GPU**: Intel GPU drivers (Arc series+)

### For External Providers
- API keys from providers you want to use (OpenAI, Anthropic, Google, etc.)
- Internet access for API calls and provider.db downloads

---

## Local Setup (Ollama)

This is the classic setup: SmarterRouter routes to models running in a local Ollama instance.

### 1. Prerequisites

Make sure Ollama is running:

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
# Or if Ollama is installed natively on your system
ollama serve
```

Pull some models:

```bash
ollama pull llama3
ollama pull codellama
```

### 2. Environment Configuration

```bash
# Copy the template
cp ENV_DEFAULT .env

# Edit .env - only these are essential for local setup:
nano .env
```

**Minimal .env for local-only:**

```bash
ROUTER_PROVIDER=ollama
ROUTER_OLLAMA_URL=http://localhost:11434
ROUTER_ADMIN_API_KEY=your-secret-key-here  # Required for production
```

### 3. Start SmarterRouter

```bash
docker-compose up -d
```

### 4. Verify

```bash
curl http://localhost:11436/health

# Should return: {"status":"healthy"}
```

### 5. Connect to OpenWebUI

1. Open OpenWebUI → **Settings** → **Connections** → **Add Connection**
2. Configure:
   - **Name:** `SmarterRouter`
   - **Base URL:** `http://localhost:11436/v1`
   - **API Key:** (leave empty)
   - **Model:** `smarterrouter/main`
3. Save and start chatting

SmarterRouter will automatically select the best model for each prompt!

---

## External Providers Only

Use this setup if you want to route to cloud APIs (OpenAI, Anthropic, etc.) without any local Ollama.

### 1. Prerequisites

Get API keys from providers:

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **Google**: https://aistudio.google.com/app/apikey
- **Cohere**: https://dashboard.cohere.com/api-keys
- **Mistral**: https://console.mistral.ai/api-keys

### 2. Create External-Only Compose File

Replace `docker-compose.yml` with `docker-compose.external.yml` (or create your own):

```yaml
# docker-compose.external.yml
version: '3.8'

services:
  smarterrouter:
    image: ghcr.io/peva3/smarterrouter:latest
    container_name: smarterrouter
    ports:
      - "11436:11436"
    environment:
      # Router settings
      - ROUTER_PROVIDER=ollama  # Keep for compatibility; won't actually connect to local
      - ROUTER_OLLAMA_URL=http://localhost:11434  # Not used but required
      - ROUTER_HOST=0.0.0.0
      - ROUTER_PORT=11436
      
      # External providers
      - ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
      - ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google
      
      # API keys (set via .env or directly)
      - ROUTER_OPENAI_API_KEY=${OPENAI_API_KEY}
      - ROUTER_ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ROUTER_GOOGLE_API_KEY=${GOOGLE_API_KEY}
      
      # provider.db auto-update
      - ROUTER_PROVIDER_DB_ENABLED=true
      - ROUTER_PROVIDER_DB_AUTO_UPDATE_HOURS=4
      
      # Production security
      - ROUTER_ADMIN_API_KEY=${ROUTER_ADMIN_API_KEY}
      
      # Volume mounts
      - ./data:/app/hubrouter/data
      - ./logs:/app/hubrouter/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11436/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Configure .env

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

ROUTER_ADMIN_API_KEY=your-secret-admin-key
```

### 4. Start

```bash
docker-compose -f docker-compose.external.yml up -d
```

### 5. Use External Models

In OpenWebUI (or any OpenAI-compatible client), select models with provider prefixes:

- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `google/gemini-1.5-pro`

Browse all available models from provider.db:
```bash
curl http://localhost:11436/v1/models | jq '.data[].id'
```

---

## Mixed Local + External

You can use both local Ollama and external providers simultaneously. This is the most flexible setup.

### 1. Prerequisites

- Ollama running with your favorite models
- API keys for any external providers you want to use

### 2. Configuration

```bash
# .env file
ROUTER_PROVIDER=ollama
ROUTER_OLLAMA_URL=http://localhost:11434

# Enable external providers
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic

# API keys
ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start

Use the regular `docker-compose.yml` - all external settings are controlled via `.env`.

### 4. Use Hybrid Models

Now you can select from both local and external models in your client:

- Local: `llama3`, `codellama`, `mistral`
- External: `openai/gpt-4o`, `anthropic/claude-3-opus`

The router will automatically pick the best model for each prompt based on capability, speed, and cost.

---

## Docker Commands

### View Logs

```bash
# Follow logs
docker-compose logs -f

# Only SmarterRouter logs
docker-compose logs -f smarterrouter

# Provider.db download logs
docker-compose logs smarterrouter | grep "provider.db"
```

### Restart

```bash
docker-compose restart
```

### Stop

```bash
docker-compose down
```

### Update

```bash
docker-compose pull
docker-compose up -d
```

### Execute Commands Inside Container

```bash
docker-compose exec smarterrouter bash

# Check provider.db stats
python -c "from router.provider_db import get_provider_db; print(get_provider_db().get_stats())"
```

---

## GPU Configuration

The `docker-compose.yml` file includes configuration for all major GPU types. Choose the section that matches your hardware.

### NVIDIA GPUs (Most Common)

**Requirements:**
- NVIDIA drivers installed on host
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

**Usage:**

```bash
# Enable GPU access with --compatibility flag
docker-compose --compatibility up -d

# OR with Docker Compose v2:
docker compose up -d
```

**Multi-GPU:**

The `deploy` section uses `count: all` to expose all GPUs. To limit to specific GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Only first 2 GPUs
          capabilities: [gpu]
```

### AMD GPUs (ROCm)

**Requirements:**
- AMD GPU with ROCm support (RX 6000/7000 series, Radeon Instinct, Radeon Pro)
- ROCm runtime installed: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html

**Configuration:**

1. **Comment out** the entire `deploy:` section (lines 56-62)
2. **Uncomment** the `devices:` section (lines 74-78)

```yaml
    # Comment out the deploy section above
    #
    # devices:
    #   - /dev/kfd              # AMD Kernel Fusion Driver
    #   - /dev/dri              # Direct Rendering Infrastructure
    # environment:
    #   - ROCM_PATH=/opt/rocm   # Optional: if using ROCm base image
```

Then start normally:

```bash
docker-compose up -d
```

### Intel Arc GPUs

**Requirements:**
- Intel Arc A-series GPU (A380, A770, etc.)
- Intel GPU drivers (i915 kernel module)

**Configuration:**

1. **Comment out** the `deploy` section
2. **Uncomment** the `devices` section (lines 89-92)

```yaml
    # devices:
    #   - /dev/dri              # Direct Rendering Infrastructure
    # environment:
    #   - LEVEL_ZERO_DEVICE=0   # Optional: For oneAPI/Level Zero
```

### CPU-Only / No GPU

1. **Comment out** all GPU configuration sections
2. **Add to your `.env`:**

```bash
ROUTER_VRAM_MONITOR_ENABLED=false
```

### Multi-GPU (NVIDIA + AMD)

1. Keep the `deploy` section for NVIDIA
2. Uncomment the `devices` section for AMD

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    devices:
      - /dev/kfd
      - /dev/dri
```

---

## Provider.db Auto-Update

provider.db is automatically updated in the background by SmarterRouter.

### Configuration

```bash
# Update frequency in hours (default: 4)
ROUTER_PROVIDER_DB_AUTO_UPDATE_HOURS=4

# Set to 0 to disable auto-updates
ROUTER_PROVIDER_DB_AUTO_UPDATE_HOURS=0

# Custom download URL (advanced)
ROUTER_PROVIDER_DB_DOWNLOAD_URL=https://custom-cdn.com/provider.db
```

### Manual Update

```bash
# Touch the database file to force re-download on next startup
docker-compose exec smarterrouter touch /app/hubrouter/data/provider.db

# Or restart the container (triggers immediate download if stale)
docker-compose restart smarterrouter
```

---

## Production Considerations

### 1. Security

**Always set admin API key:**

```bash
ROUTER_ADMIN_API_KEY=generate-a-strong-random-key-here
```

**Secure the .env file:**

```bash
chmod 600 .env
```

**Use HTTPS reverse proxy:**

```nginx
# nginx config snippet
location / {
    proxy_pass http://localhost:11436;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

### 2. Persistence

Mount persistent volumes:

```yaml
volumes:
  - ./data:/app/hubrouter/data  # Contains provider.db and SQLite DBs
  - ./logs:/app/hubrouter/logs  # Application logs
```

### 3. Monitoring

**Health endpoint:**

```bash
curl http://localhost:11436/health
```

**Admin metrics (requires ROUTER_ADMIN_API_KEY):**

```bash
curl -H "Authorization: Bearer YOUR_ADMIN_KEY" http://localhost:11436/admin/stats
```

**VRAM monitoring:**

```bash
curl -H "Authorization: Bearer YOUR_ADMIN_KEY" http://localhost:11436/admin/vram
```

### 4. Resource Limits

Limit container resources to prevent runaway usage:

```yaml
services:
  smarterrouter:
    # ... other config
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

### 5. Network

For production, consider:

- Using a custom network
- Restricting access to trusted IPs
- Adding rate limiting via reverse proxy

### 6. Logging

Configure Docker logging driver for persistent logs:

```yaml
services:
  smarterrouter:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs smarterrouter

# Common issues:
# - Port 11436 already in use
# - .env file missing or wrong permissions
# - GPU drivers not accessible (check /dev/kfd, /dev/dri)
```

### provider.db not downloading

```bash
# Check logs for errors
docker-compose logs smarterrouter | grep provider.db

# Manually trigger download
docker-compose exec smarterrouter python -c "from main import download_provider_db; import asyncio; asyncio.run(download_provider_db())"
```

### External provider API errors

Make sure API keys are set correctly:

```bash
docker-compose exec smarterrouter printenv | grep API_KEY
```

Test connectivity:

```bash
docker-compose exec smarterrouter curl -H "Authorization: Bearer YOUR_KEY" https://api.openai.com/v1/models
```

### GPU not detected

```bash
# Check what GPUs Docker can see
docker-compose exec smarterrouter nvidia-smi  # NVIDIA
docker-compose exec smarterrouter rocm-smi   # AMD

# If not accessible, check host device permissions
ls -la /dev/kfd /dev/dri
```

---

## Advanced: Custom Dockerfile

If you need custom modifications, create your own Dockerfile:

```dockerfile
FROM ghcr.io/peva3/smarterrouter:latest

# Install additional tools
USER root
RUN apt-get update && apt-get install -y \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy custom config
COPY custom-config.yaml /app/hubrouter/config.yaml

USER smarterrouter
```

Then in `docker-compose.yml`:

```yaml
services:
  smarterrouter:
    build: .
    # ... rest of config
```

---

## See Also

- [Configuration Reference](../docs/configuration.md) - All environment variables
- [External Providers Guide](../docs/external-providers.md) - Setting up cloud providers
- [Performance Tuning](../docs/performance.md) - Optimize for your workload
- [Troubleshooting](../docs/troubleshooting.md) - Common issues and solutions
