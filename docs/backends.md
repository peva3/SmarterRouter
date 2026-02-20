# Backend Providers

SmarterRouter supports multiple LLM backends through a unified interface.

## Comparison

| Feature | Ollama | llama.cpp | OpenAI-Compatible |
|---------|--------|-----------|-------------------|
| **Local inference** | ✅ Native | ✅ Server | ⚠️ Via proxy |
| **VRAM management** | ✅ Full | ⚠️ Partial | ❌ None |
| **Model unloading** | ✅ Yes | ⚠️ Graceful no-op | ❌ No |
| **Embeddings** | ✅ Yes | ✅ Yes | ✅ Via API |
| **Best for** | Local production | High-performance servers | External APIs |

---

## Ollama (Default, Recommended)

### Setup

1. [Install Ollama](https://ollama.ai/download)
2. Pull models:
   ```bash
   ollama pull llama3:70b
   ollama pull codellama:34b
   ollama pull phi3:mini
   ```
3. Start Ollama service:
   ```bash
   systemctl --user start ollama
   # or
   ollama serve
   ```

### Configuration

```env
ROUTER_PROVIDER=ollama
ROUTER_OLLAMA_URL=http://localhost:11434
```

### Advantages

- **Native integration** - No API translation layer
- **Full VRAM management** - SmarterRouter can load/unload models dynamically
- **Embeddings support** - Uses Ollama's native `/api/embedd` endpoint
- **Production-ready** - Stable, well-tested

### Notes

- Ollama must be running before SmarterRouter starts
- Models are discovered automatically via `/api/tags`
- VRAM monitoring uses `nvidia-smi` to measure actual GPU usage

---

## llama.cpp Server

### Setup

1. Build or download [llama.cpp](https://github.com/ggerganov/llama.cpp)
2. Start server:
   ```bash
   ./server -m models/llama3-70b.gguf -c 4096 --port 8080
   ```
3. Add models by starting additional server instances or using llama-swap

### Configuration

```env
ROUTER_PROVIDER=llama.cpp
ROUTER_OLLAMA_URL=http://localhost:8080  # llama.cpp server URL
```

### Advantages

- **High performance** - Direct GGUF execution, no Docker overhead
- **Flexible deployment** - Can run on CPU or GPU
- **Multiple backends** - Works with llama-swap for dynamic model switching

### Limitations

- **No explicit model unloading** - llama.cpp server loads models into memory; unloading returns `False` gracefully but models stay loaded
- **Manual model management** - You manage server instances; SmarterRouter can't load/unload dynamically

### Tips

- Use llama-swap to dynamically switch models on same server
- Allocate sufficient context buffer: `-c 8192` for long conversations
- Use `--threads` and `--gpu-layers` to optimize performance

---

## OpenAI-Compatible API

Works with any service that implements OpenAI's API spec.

### Compatible Services

- **OpenAI** (`https://api.openai.com/v1`)
- **Anthropic** (via [anthropic-openai](https://github.com/anthropics/anthropic-sdk-python#openai-compatibility) or [LiteLLM](https://docs.litellm.ai/docs/providers/anthropic))
- **vLLM** (self-hosted)
- **Text Generation Inference (TGI)**
- **LiteLLM Proxy** - Tried for 100+ providers
- **LocalAI**
- **Ollama with OpenAI compatibility** (`OLLAMA_ORIGINS=*`)

### Configuration

```env
ROUTER_PROVIDER=openai
ROUTER_OPENAI_BASE_URL=https://api.openai.com/v1
ROUTER_OPENAI_API_KEY=sk-your-key-here
```

### Example: OpenRouter (Access to 100+ models)

```env
ROUTER_PROVIDER=openai
ROUTER_OPENAI_BASE_URL=https://openrouter.ai/api/v1
ROUTER_OPENAI_API_KEY=sk-or-v1-your-key
ROUTER_MODEL_PREFIX=  # leave empty, model names already include provider
```

Now you can route between OpenAI, Anthropic, and other providers through a single SmarterRouter instance!

### Example: Together AI

```env
ROUTER_PROVIDER=openai
ROUTER_OPENAI_BASE_URL=https://api.together.xyz/v1
ROUTER_OPENAI_API_KEY=your-key
```

### Advantages

- **Universal compatibility** - Any OpenAI-compatible endpoint works
- **Multi-provider routing** - Route between OpenAI, Anthropic, etc.
- **Cloud scale** - No local VRAM constraints

### Limitations

- **No local VRAM management** - Cloud APIs manage their own resources
- **API costs** - Pay per token
- **Rate limits** - Subject to provider limits
- **No model unloading** - Not applicable

---

## Testing Backend Compatibility

All backends have comprehensive test suites:

```bash
# Run backend-specific tests
pytest tests/test_ollama_backend.py -v
pytest tests/test_llama_cpp_backend.py -v
pytest tests/test_openai_backend.py -v

# Run contract tests (ensures all backends behave consistently)
pytest tests/test_backend_contract.py -v
```

---

## Backend Troubleshooting

### "All connection attempts failed"

1. Verify backend is running and accessible:
   ```bash
   curl http://localhost:11434/api/tags  # adjust port
   ```
2. Check firewall rules if remote
3. Verify `ROUTER_OLLAMA_URL` is correct
4. Check Docker networking (use `host.docker.internal` or `172.17.0.1`)

---

### Model not appearing in discovery

- Ensure model is pulled/loaded in backend
- Check backend's model list endpoint manually
- Restart SmarterRouter after adding new models

---

### Slow responses with llama.cpp

- Check context size: `-c 4096` or higher recommended
- Enable GPU layers if available: `--gpu-layers 100`
- Use quantized models (GGUF) for faster CPU inference

---

### OpenAI API rate limits

- Check provider dashboard for usage
- Add retry logic in your application (SmarterRouter doesn't retry failed API calls)
- Consider adding multiple API keys for load balancing (coming soon)

---

## Future Backends

The backend abstraction layer makes adding new providers straightforward. Potential future additions:
- HuggingFace Text Generation Inference
- AWS Bedrock
- Google Vertex AI
- Azure OpenAI
- Custom RPC protocols

If you need a specific backend, [open an issue](https://github.com/peva3/SmarterRouter/issues).
