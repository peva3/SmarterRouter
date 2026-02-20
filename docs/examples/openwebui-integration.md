# OpenWebUI Integration

Complete guide to integrating SmarterRouter with OpenWebUI (formerly Ollama WebUI).

## Prerequisites

- SmarterRouter running (Docker or manual)
- OpenWebUI installed and running
- Both on same network (or same host)

## Steps

### 1. Start SmarterRouter

If not already running:

```bash
cd /path/to/SmarterRouter
docker-compose up -d
```

Verify: `curl http://localhost:11436/health` should return `{"status":"healthy"}`

### 2. Open OpenWebUI Settings

1. Open your OpenWebUI in browser (default: `http://localhost:3000`)
2. Click **Settings** (gear icon) → **Connections**
3. Click **Add a Connection**

### 3. Configure Connection

Fill in the form:

| Field | Value |
|-------|-------|
| **Name** | SmarterRouter |
| **Base URL** | `http://localhost:11436/v1` |
| **API Key** | (leave blank - SmarterRouter doesn't require chat API key) |
| **Model** | `smarterrouter/main` (select from dropdown) |

Click **Save**.

### 4. Test It

1. Return to **Chat** tab
2. Create new conversation
3. Model should default to `SmarterRouter` (or select from dropdown)
4. Send a test message: "Write a Python function to calculate fibonacci numbers"

### 5. Verify Routing

After receiving response, you'll see a signature at the bottom:
```
Model: llama3:70b
```

That indicates which model SmarterRouter selected. Try different prompts to see routing behavior:

- **Coding prompt:** "Explain how context managers work in Python" → Should route to coder model
- **Reasoning prompt:** "If a train leaves at 2pm traveling 60mph..." → Should route to reasoning model
- **Simple prompt:** "What's the weather today?" → Should route to small fast model

### 6. Optional: Adjust Quality Preference

If you want faster but lower quality responses, adjust in SmarterRouter `.env`:

```bash
ROUTER_QUALITY_PREFERENCE=0.3  # favor speed
```

Then restart: `docker-compose restart`

---

## Advanced Configuration

### Enable Admin API Access

To access admin features (profiles, VRAM monitoring) from OpenWebUI:

1. Set admin API key in SmarterRouter `.env`:
   ```bash
   ROUTER_ADMIN_API_KEY=sk-secure-random-key
   ```
   Generate with: `openssl rand -hex 32`

2. Restart SmarterRouter: `docker-compose restart`

3. In OpenWebUI (if admin panel available), configure admin endpoint with this key.

### Pinning a Fast Model

For instant responses to simple queries, pin a small model:

```bash
# In .env
ROUTER_PINNED_MODEL=phi3:mini
```

Restart SmarterRouter. The pinned model stays loaded in VRAM, providing sub-second responses for simple prompts.

---

## Troubleshooting

### "Connection refused" or timeout

- Verify SmarterRouter is running: `curl http://localhost:11436/health`
- Check that ports match (`11436` default)
- If using Docker, ensure both containers on same network

---

### Model not showing in dropdown

- SmarterRouter exposes only `smarterrouter/main` as the model
- Make sure you selected "SmarterRouter" connection
- Refresh OpenWebUI page

---

### Slow first response

- First request to a model triggers loading (cold start)
- Subsequent requests to same model will be faster
- Consider pinning a small model: `ROUTER_PINNED_MODEL=phi3:mini`

---

### "All models failed"

- Check SmarterRouter health: `curl http://localhost:11436/health`
- Verify backend (Ollama) is running: `curl http://localhost:11434/api/tags`
- Check logs: `docker logs smarterrouter`

---

## Direct API Testing

While OpenWebUI works through its UI, you can also test directly:

```bash
# Chat completion
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Get info
curl http://localhost:11436/v1/models

# Health check
curl http://localhost:11436/health
```

Use these for debugging before connecting OpenWebUI.

---

## Next Steps

- [Configuration Reference](../configuration.md) - Fine-tune SmarterRouter behavior
- [Performance Tuning](../performance.md) - Optimize for your workload
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
- [API Documentation](../api.md) - Complete API reference
