# SmarterRouter

**Intelligent, multi-backend AI router** that sits between your application and various LLM providers. It profiles your models, aggregates benchmark data, and intelligently routes each query to the best available model for the task—all locally, all free.
**Key Benefits:**
- **Zero manual model selection** - AI automatically picks the right model for each prompt
- **All local, zero cost** - No cloud API fees, works with your existing models
- **Production-ready** - Monitoring, metrics, and error handling built-in
- **Drop-in replacement** - Works with any OpenAI-compatible client

  
## Why SmarterRouter? (vs other LLM proxies)

| Feature | SmarterRouter | OptiLLM | ClewdR | LLM-API-Proxy | Reader |
|---------|---------------|---------|--------|---------------|---------|
| **Intelligent Routing** | ✅ Auto-selects best model | ❌ Manual config | ❌ Claude-only | ❌ Manual routing | ❌ URL-only |
| **Multi-Backend Support** | ✅ Ollama + llama.cpp + OpenAI | ❌ OpenAI-only | ❌ Claude-only | ✅ 100+ providers | ❌ URL proxy |
| **Local-first** | ✅ All local models | ⚠️ Cloud proxy | ⚠️ Cloud proxy | ⚠️ Cloud proxy | ⚠️ Cloud proxy |
| **Zero Code Changes** | ✅ OpenAI-compatible | ✅ OpenAI-compatible | ✅ OpenAI-compatible | ✅ OpenAI-compatible | ✅ URL proxy |
| **Production Features** | ✅ Monitoring + Metrics | ✅ Metrics | ✅ Dashboard | ✅ Resilience | ✅ Simple |
| **Learning Capability** | ✅ Profiles models over time | ❌ Static config | ❌ Static config | ❌ Static config | ❌ Static config |



---

## Quick Start (5 minutes)

Get up and running with Docker in three commands:

```bash
# 1. Clone the repository
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter

# 2. Start with Docker Compose
docker-compose up -d

# 3. Verify it's running
curl http://localhost:11436/health
```

That's it! SmarterRouter will:
- ✅ Discover all your Ollama models automatically
- ✅ Profile each model for performance on your hardware (first run takes 30-60 min)
- ✅ Start routing queries to the best model

**Access the router at:** `http://localhost:11436`

### Connect to OpenWebUI

1. Open OpenWebUI → **Settings** → **Connections** → **Add Connection**
2. Configure:
   - **Name:** `SmarterRouter`
   - **Base URL:** `http://localhost:11436/v1`
   - **API Key:** (leave empty)
   - **Model:** `smarterrouter/main`
3. Save and start chatting

SmarterRouter will automatically select the best model for each prompt!

---

## What Gets Automated?

- ✅ **Model discovery** - Automatically finds all available models from your backend
- ✅ **Performance profiling** - Tests each model with standardized prompts on your hardware
- ✅ **Smart routing** - Analyzes prompts and picks the optimal model based on category and complexity
- ✅ **VRAM management** - Auto-detects all GPUs (NVIDIA, AMD, Intel, Apple Silicon), monitors usage, and unloads models when needed
- ✅ **Fallback handling** - Automatically retries with backup models if primary fails
- ✅ **Response caching** - Caches identical prompts for instant responses
- ✅ **Continuous learning** - Collects user feedback to improve routing decisions

---

## Configuration Basics

All configuration is via the `.env` file. Copy the template and customize:

```bash
cp ENV_DEFAULT .env
nano .env  # edit as needed
```

**Essential settings:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `ROUTER_OLLAMA_URL` | Your backend URL | `http://localhost:11434` |
| `ROUTER_PROVIDER` | Backend type: ollama, llama.cpp, openai | `ollama` |
| `ROUTER_QUALITY_PREFERENCE` | 0.0 (speed) to 1.0 (quality) | `0.5` |
| `ROUTER_PINNED_MODEL` | Keep a small model always loaded (optional) | (none) |
| `ROUTER_ADMIN_API_KEY` | **Required for production** to secure admin endpoints | (none) |

**VRAM monitoring:** Enabled by default with auto-detection across NVIDIA, AMD, Intel, and Apple Silicon GPUs. Multi-GPU systems are fully supported. See [Configuration Reference](docs/configuration.md#vram-monitoring) for details.

**⚠️ Production security:** Always set `ROUTER_ADMIN_API_KEY` in production to protect admin endpoints.

For complete configuration reference, see [docs/configuration.md](docs/configuration.md).

---

## Documentation

**Getting Started:**
- [Installation Guide](docs/installation.md) - Detailed setup instructions for all platforms
- [Configuration Reference](docs/configuration.md) - Complete environment variable reference
- [API Documentation](docs/api.md) - All endpoints and request/response formats

**In-Depth Guides:**
- [Performance Tuning](docs/performance.md) - Optimize for latency, throughput, or quality
- [Backend Providers](docs/backends.md) - Set up Ollama, llama.cpp, or OpenAI-compatible backends
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

**Examples:**
- [OpenWebUI Integration](docs/examples/openwebui-integration.md) - Step-by-step connection guide
- [Production Deployment](docs/examples/production-deployment.md) - Security, monitoring, HA
- [Client Integration](docs/examples/client-integration.md) - Using SmarterRouter with various tools

**Want to see how the sausage is made?**
- [DEEPDIVE.md](DEEPDIVE.md) - Architecture, design decisions, and implementation details for the technically curious

**Other Files:**
- [CHANGELOG.md](CHANGELOG.md) - Version history and releases
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [SECURITY.md](SECURITY.md) - Security policy and vulnerability reporting
- [ENV_DEFAULT](ENV_DEFAULT) - Full list of configuration options

---

## Need Help?

- **Bugs & Feature Requests:** [GitHub Issues](https://github.com/peva3/SmarterRouter/issues)
- **Discussions & Questions:** [GitHub Discussions](https://github.com/peva3/SmarterRouter/discussions)
- **Security Issues:** See [SECURITY.md](SECURITY.md)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
