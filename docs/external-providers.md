# External Provider Setup

This guide shows you how to configure SmarterRouter to route to external cloud providers (OpenAI, Anthropic, Google, etc.) alongside your local Ollama models.

## Overview

External provider support consists of two parts:

1. **provider.db** - Benchmark database with scores for 400+ models
2. **External API Integration** - Actually route requests to external providers

## Supported Providers

| Provider | Model Prefix | API Key | Base URL |
|----------|--------------|---------|----------|
| OpenAI | `openai/` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| Google | `google/` | `GOOGLE_API_KEY` | `https://generativelanguage.googleapis.com/v1` |
| Cohere | `cohere/` | `COHERE_API_KEY` | `https://api.cohere.ai/v1` |
| Mistral | `mistral/` | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |

## Quick Start

```bash
# 1. Edit your .env file
nano .env

# Add these settings:
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google
ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...

# 2. Restart SmarterRouter
docker-compose restart

# 3. Use external models
# In OpenWebUI, select models with prefix: openai/gpt-4o, anthropic/claude-3-opus
```

## Configuration

All configuration goes in your `.env` file:

```bash
# Enable external provider routing
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true

# List of providers to use (comma-separated)
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google

# API Keys (at least one required)
ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
ROUTER_GOOGLE_API_KEY=...
ROUTER_COHERE_API_KEY=...
ROUTER_MISTRAL_API_KEY=...

# Optional: Custom base URLs (for proxies/self-hosted)
ROUTER_ANTHROPIC_BASE_URL=https://custom-endpoint.com
ROUTER_GOOGLE_BASE_URL=https://custom-endpoint.com
```

**Note:** provider.db is used automatically for benchmark data. No additional configuration needed - it downloads to `data/provider.db` and updates every 4 hours.

## Model Naming

Use provider prefixes to identify external models:

- `openai/gpt-4o` - OpenAI's GPT-4o
- `openai/gpt-4-turbo` - OpenAI's GPT-4 Turbo
- `anthropic/claude-3-opus` - Anthropic's Claude 3 Opus
- `anthropic/claude-3-sonnet` - Anthropic's Claude 3 Sonnet
- `google/gemini-1.5-pro` - Google's Gemini 1.5 Pro
- `cohere/command-r-plus` - Cohere's Command R+
- `mistral/mistral-large` - Mistral's Large model

Models are automatically discovered from provider.db (400+ models available).

## How Routing Works

1. **Benchmark Data** - provider.db provides scores for reasoning, coding, and general knowledge
2. **Hybrid Selection** - Router combines:
   - Local model profiling (if model is also available locally)
   - External benchmark scores from provider.db
   - Model capabilities (vision, tool calling)
3. **External API Call** - When an external model is selected, BackendRegistry routes to the appropriate provider

## Mixing Local and External

You can use both local Ollama and external providers simultaneously:

```bash
# Enable both
ROUTER_PROVIDER=ollama  # Keep local backend
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic
```

In your client, you can:
- Use `llama3` (local), `openai/gpt-4o` (external), `anthropic/claude-3-opus` (external)
- SmarterRouter will automatically pick the best model for each prompt

## Docker Setup (External Only)

For setups that only use external providers (no local Ollama):

```yaml
# docker-compose.external.yml
version: '3.8'
services:
  smarterrouter:
    image: smarterrouter:latest
    container_name: smarterrouter
    ports:
      - "11436:11436"
    environment:
      - ROUTER_PROVIDER=ollama  # Keep for compatibility, but won't connect
      - ROUTER_OLLAMA_URL=http://localhost:11434  # Won't be used
      - ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
      - ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google
      - ROUTER_OPENAI_API_KEY=${OPENAI_API_KEY}
      - ROUTER_ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ROUTER_GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./data:/app/hubrouter/data
      - ./logs:/app/hubrouter/logs
    restart: unless-stopped
```

Run with:

```bash
docker-compose -f docker-compose.external.yml up -d
```

## Provider.db Auto-Update

provider.db is automatically updated every 4 hours by default. You can customize:

```bash
# Update frequency (hours, 0 = disabled)
ROUTER_PROVIDER_DB_AUTO_UPDATE_HOURS=4

# Download URL (advanced users)
ROUTER_PROVIDER_DB_DOWNLOAD_URL=https://raw.githubusercontent.com/peva3/smarterrouter-provider/refs/heads/main/data/provider.db
```

## Troubleshooting

### "No external backend configured" error

Make sure:
- `ROUTER_EXTERNAL_PROVIDERS_ENABLED=true`
- The provider (e.g., `openai`) is in `ROUTER_EXTERNAL_PROVIDERS`
- The corresponding API key is set

### Model not found in provider.db

Check that provider.db is present:

```bash
ls -la data/provider.db
# Should see ~200KB file
```

If missing, the system will fall back to local models only or keyword-based routing.

### Performance issues with external providers

External API latency can be higher than local models. Consider:
- Increasing `ROUTER_GENERATION_TIMEOUT` (default 120s)
- Using faster models for simple tasks
- Keeping some local models as fallback

## Security Considerations

**API Key Storage:**
- Store API keys in `.env` file (not committed to git)
- Use Docker secrets or vault solutions for production
- Set proper file permissions: `chmod 600 .env`

**Network Security:**
- Use HTTPS endpoints (all default URLs are HTTPS)
- Consider using a proxy/VPN for additional privacy
- Monitor API usage for unexpected activity

## Advanced Configuration

### Custom Base URLs

Use with reverse proxies or self-hosted providers:

```bash
ROUTER_ANTHROPIC_BASE_URL=http://localhost:8080/v1
ROUTER_EXTERNAL_PROVIDERS=anthropic
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
```

The endpoint must be OpenAI-compatible (same API format).

### Model Prefix Customization

If your external provider uses a different URL pattern, you can adjust the model prefix in `router/backends/external.py`. Most providers work with the defaults.

## See Also

- [Configuration Reference](configuration.md) - All environment variables
- [Backend Providers](backends.md) - Adding new backend types
- [Performance Tuning](performance.md) - Optimize routing decisions
