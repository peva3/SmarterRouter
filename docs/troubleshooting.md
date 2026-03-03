# Troubleshooting Guide

## Table of Contents
- [Common Issues](#common-issues)
- [Diagnosis Steps](#diagnosis-steps)
- [Performance Problems](#performance-problems)
- [Database Issues](#database-issues)
- [Docker Issues](#docker-issues)
- [GPU Issues (All Vendors)](#gpu-issues-all-vendors)
- [Debug Mode](#debug-mode)

> **GPU Support:** SmarterRouter supports NVIDIA, AMD (ROCm), Intel Arc, and Apple Silicon GPUs. If your GPU isn't detected, check the GPU issues section below for vendor-specific troubleshooting.

## Common Issues

### "All connection attempts failed" / "Failed to list models"

**Symptom:** SmarterRouter cannot reach your LLM backend (Ollama, llama.cpp, etc.).

**Diagnosis:**
```bash
curl http://localhost:11434/api/tags  # adjust port if needed
```

Should return:
```json
{
  "models": [...]
}
```

**Solutions:**
1. Verify backend is running: `systemctl status ollama` or `docker ps`
2. Check `ROUTER_OLLAMA_URL` in `.env` matches backend URL
3. Test connectivity from SmarterRouter container:
   ```bash
   docker exec smarterrouter curl http://<backend-ip>:<port>/api/tags
   ```
4. For Docker networking:
   - Host-only: use `host.docker.internal` (Mac/Windows) or `172.17.0.1` (Linux)
   - Docker Compose: use service name (e.g., `ollama:11434`)

---

### "Port already in use"

**Symptom:** Container fails to start; logs show "address already in use"

**Solution:**
```bash
# Check what's using the port
lsof -i :11436
# or
netstat -tulpn | grep 11436

# Option 1: Stop existing container
docker stop smarterrouter && docker rm smarterrouter

# Option 2: Change port in .env or docker-compose.yml
ROUTER_PORT=11437
```

---

### "Database path is a directory"

**Symptom:** Error "CRITICAL: Database path is a directory"

**Cause:** Docker created a directory instead of a file because the path didn't exist before mounting.

**Solution:**
```bash
# Stop container
docker-compose down

# Remove the erroneous directory
rm -rf router.db  # or data/router.db depending on your setup

# Ensure your docker-compose.yml has correct volume mapping:
# - ./router.db:/app/router.db  (single file)
# OR
# - ./data:/app/data            (directory, database inside)

# Restart
docker-compose up -d
```

---

## GPU Issues (All Vendors)

SmarterRouter auto-detects GPUs from all vendors on startup. Check logs for detection messages.

### "nvidia-smi not found"

**Symptom:** GPU monitoring disabled; VRAM not tracking.

**Solution:**
1. Install NVIDIA drivers on host
2. Install NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```
3. Test:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```
4. Restart SmarterRouter with `--gpus all` or `--compatibility`

---

### "AMD APU shows wrong VRAM (e.g., 7.7GB instead of 58GB)"

**Symptom:** AMD APU (Ryzen AI, Radeon 800M series) detected with small VRAM instead of full unified memory.

**Diagnosis:**
```bash
# Check what's being reported
cat /sys/class/drm/card*/device/mem_info_vram_total
cat /sys/class/drm/card*/device/mem_info_gtt_total

# VRAM total should be small (< 8GB) for APUs
# GTT total should be large (~system RAM) for APUs
```

**Explanation:**
APUs have two memory pools:
- **VRAM** (mem_info_vram_*): Small BIOS carve-out (512MB-8GB)
- **GTT** (mem_info_gtt_*): Unified memory pool (actual usable GPU memory)

SmarterRouter auto-detects APUs (VRAM < 4GB) and uses GTT for total memory.

**Solutions:**
1. **Check BIOS UMA Buffer** - Should be set to minimum (512MB-2GB):
   - Large UMA buffer wastes RAM and confuses detection
   - GTT pool is the real usable memory, not VRAM

2. **Manual Override** if detection still fails:
   ```bash
   # In .env
   ROUTER_AMD_UNIFIED_MEMORY_GB=58  # ~90% of your RAM
   ```

3. **Verify ROCm architecture** (for gfx1150/gfx1151):
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=11.5.1
   ```

---

### "AMD GPU not detected"

**Symptom:** AMD GPU present but VRAM monitoring disabled.

**Diagnosis:**
```bash
# Check for rocm-smi
rocm-smi

# Check sysfs entries
ls /sys/class/drm/card*/device/mem_info_vram_total
```

**Solutions:**
1. Install ROCm runtime:
   ```bash
   # Ubuntu 22.04
   sudo amdgpu-install --usecase=rocm,graphics --no-dkms
   sudo usermod -a -G render,video $USER
   ```
2. For Docker, ensure device passthrough:
   ```yaml
   devices:
     - /dev/kfd
     - /dev/dri
   ```
3. Verify AMD vendor ID in sysfs:
   ```bash
   cat /sys/class/drm/card*/device/vendor
   # Should show: 0x1002 for AMD
   ```

---

### "Intel Arc GPU not detected"

**Symptom:** Intel Arc GPU present but VRAM monitoring disabled.

**Diagnosis:**
```bash
# Check for Intel GPU with dedicated memory
ls /sys/class/drm/card*/device/lmem_total

# Check driver loaded
lsmod | grep i915
```

**Solutions:**
1. Ensure Intel GPU drivers are installed (kernel 5.19+ recommended)
2. For Docker, ensure device passthrough:
   ```yaml
   devices:
     - /dev/dri
   ```
3. Verify Intel vendor ID:
   ```bash
   cat /sys/class/drm/card*/device/vendor
   # Should show: 0x8086 for Intel
   ```

**Note:** Only Intel Arc dedicated GPUs with local memory (lmem) are supported. Integrated Intel UHD/Iris GPUs use shared system memory and are not monitored.

---

### "Apple Silicon GPU memory incorrect"

**Symptom:** Apple Silicon detected but VRAM estimate is wrong.

**Solutions:**
1. Manually set unified memory:
   ```bash
   # In .env
   ROUTER_APPLE_UNIFIED_MEMORY_GB=16  # for 16GB Mac
   ```
2. Apple Silicon VRAM is estimated as 75% of total RAM by default
3. Run SmarterRouter natively on macOS host (not in Docker) for accurate detection:
   ```bash
   system_profiler SPHardwareDataType | grep "Memory:"
   ```

**Important:** Docker Desktop on macOS cannot pass GPU to containers. You must run SmarterRouter on the host directly.

---

### First request is very slow (30+ seconds)

**Causes:**
- Model cold-start: First time loading a model into GPU memory
- VRAM pressure triggering model unload/reload

**Solutions:**
1. Pin a small model for fast responses:
   ```bash
   ROUTER_PINNED_MODEL=phi3:mini
   ```
2. Increase VRAM allocation if possible:
   ```bash
   ROUTER_VRAM_MAX_TOTAL_GB=<your-gpu-total-minus-2>
   ```
3. Accept that first request is always slower; subsequent requests to same model will be fast

---

### "All models failed" error

**Diagnosis:**
1. Check backend health:
   ```bash
   curl http://localhost:11434/api/tags
   ```
2. Check SmarterRouter logs:
   ```bash
   docker logs smarterrouter
   ```
3. Look for:
   - VRAM OOM errors
   - Model loading failures
   - Backend connection timeouts
4. Check VRAM usage:
   ```bash
   curl http://localhost:11436/admin/vram
   ```

**Common fixes:**
- Reduce `ROUTER_VRAM_MAX_TOTAL_GB` if set too high
- Enable auto-unload: `ROUTER_VRAM_AUTO_UNLOAD_ENABLED=true`
- Free up GPU memory (stop other containers/processes)
- Use smaller models

---

### Low-quality routing / Wrong model selected

**Diagnosis:**
1. Check model profiles:
   ```bash
   curl http://localhost:11436/admin/profiles
   ```
   - Are scores reasonable (0.0-1.0)?
   - Are all models 0.0? → Profiling failed

2. Reprofile if needed:
   ```bash
   curl -X POST "http://localhost:11436/admin/reprofile?force=true" \
     -H "Authorization: Bearer your-admin-key"
   ```

3. Use explain endpoint to understand routing decision:
   ```bash
   curl "http://localhost:11436/admin/explain?prompt=Your prompt here"
   ```

4. Check complexity detection:
   - Simple prompts → smaller models preferred
   - Complex prompts → larger models required
   - Adjust `ROUTER_QUALITY_PREFERENCE` if always too aggressive/Conservative

---

### Profiling takes too long or times out

**Cause:** Small `ROUTER_PROFILE_TIMEOUT` for large models.

**Solution:**
1. Increase timeout in `.env`:
   ```bash
   ROUTER_PROFILE_TIMEOUT=180  # 3 minutes per prompt
   ```
2. For very large models (14B+), consider:
   ```bash
   ROUTER_PROFILE_TIMEOUT=300  # 5 minutes per prompt
   ```
3. Adaptive profiling should handle this automatically; check logs for timeout errors

---

### High memory usage / Out of Memory

**Solutions:**
1. Unload unused models:
   ```bash
   # Manual trigger
   curl -X POST http://localhost:11436/admin/cache/invalidate?all=true
   ```
2. Enable aggressive auto-unload:
   ```bash
   ROUTER_VRAM_AUTO_UNLOAD_ENABLED=true
   ROUTER_VRAM_UNLOAD_THRESHOLD_PCT=75
   ROUTER_VRAM_UNLOAD_STRATEGY=largest  # free most memory first
   ```
3. Reduce `ROUTER_VRAM_MAX_TOTAL_GB` to be more conservative
4. Use smaller models
5. Increase system swap (not ideal but prevents crashes)

---

### Model not appearing in `/v1/models`

**Causes:**
- Model not discovered by backend
- Backend not running or accessible
- Profiling not complete

**Check:**
```bash
# 1. Backend connection
curl http://localhost:11434/api/tags

# 2. SmarterRouter logs for discovery
docker logs smarterrouter | grep -i "discover\|profile"

# 3. Router health
curl http://localhost:11436/health
```

**Fix:** Ensure backend is running and `ROUTER_OLLAMA_URL` is correct.

---

### Response signature showing wrong model

**Symptom:** Response says "Model: llama3:70b" but you expected a different model.

**Explanation:** This is correct behavior! SmarterRouter selected that model based on routing algorithm. To see why:
```bash
curl "http://localhost:11436/admin/explain?prompt=your prompt"
```

---

### Stale routing decisions (same model always selected)

**Cause:** Routing cache preventing re-evaluation.

**Solution:** Invalidate cache:
```bash
curl -X POST http://localhost:11436/admin/cache/invalidate?all=true
```

Or disable caching temporarily:
```bash
ROUTER_CACHE_ENABLED=false
```

---

## Performance Problems

### Slow Response Times

**Checklist:**
1. **Cold start?** First request to a model is slow while it loads. Subsequent requests faster.
2. **VRAM pressure?** Check `/admin/vram` - if utilization >90%, models may be unloading/reloading.
3. **Cache enabled?** `ROUTER_CACHE_ENABLED=true` speeds up repeat requests.
4. **Pinned model?** Set `ROUTER_PINNED_MODEL=phi3:mini` for instant responses to simple queries.
5. **Profiling complete?** If profiles are missing or 0.0 scores, router may be making poor decisions.

**Quick wins:**
```bash
# Pin a small model
ROUTER_PINNED_MODEL=phi3:mini

# Increase VRAM limit
ROUTER_VRAM_MAX_TOTAL_GB=22.0

# Enable caching
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_MAX_SIZE=1000
```

---

### High VRAM Usage

**Diagnose:**
```bash
curl http://localhost:11436/admin/vram | jq .
```

Look for:
- `loaded_models` array showing which models are in memory
- `utilization_pct`接近 100%
- `warnings` array

**Fix:**
1. Enable auto-unload (if not already):
   ```bash
   ROUTER_VRAM_AUTO_UNLOAD_ENABLED=true
   ROUTER_VRAM_UNLOAD_THRESHOLD_PCT=80
   ROUTER_VRAM_UNLOAD_STRATEGY=largest
   ```
2. Manually unload specific models via admin API (coming soon) or restart
3. Reduce `ROUTER_VRAM_MAX_TOTAL_GB` to be more conservative
4. Use smaller models (7B instead of 70B)
5. Quantize models (Q4_K_M, Q5_K_S)

---

## Docker Issues

### Container exits immediately

**Check logs:**
```bash
docker logs smarterrouter
```

**Common causes:**
- Invalid `.env` syntax (use quotes for values with spaces)
- Missing required files
- Port conflict (11436 already in use)
- Backend not accessible at startup

**Fix:** Read error message from logs; adjust configuration accordingly.

---

### Container can't find Ollama

**Symptoms:** "Failed to list models: All connection attempts failed"

**Solutions:**
1. **Both on host (not Docker):**
   ```bash
   ROUTER_OLLAMA_URL=http://localhost:11434
   ```

2. **SmarterRouter in Docker, Ollama on host (Linux):**
   ```bash
   ROUTER_OLLAMA_URL=http://172.17.0.1:11434
   ```

3. **Both in Docker Compose:**
   ```yaml
   services:
     ollama:
       image: ollama/ollama
       ports:
         - "11434:11434"
     smarterrouter:
       build: .
       environment:
         - ROUTER_OLLAMA_URL=http://ollama:11434  # use service name
   ```

4. **Test connectivity from inside container:**
   ```bash
   docker exec smarterrouter curl http://172.17.0.1:11434/api/tags
   ```

---

### Volume mount not persisting data

**Symptom:** Database resets on container restart.

**Check:**
```bash
# Verify volume is mounted
docker inspect smarterrouter | grep -A 10 "Mounts"
```

**Ensure docker-compose.yml has:**
```yaml
services:
  smarterrouter:
    volumes:
      - ./router.db:/app/router.db  # or ./data:/app/data
```

**Note:** If `router.db` doesn't exist on host initially, Docker will create it as a file (good). If parent directory structure doesn't exist, ensure it's created or use absolute path.

---

## Debug Mode

Enable verbose logging for investigation:

```bash
# In .env
ROUTER_LOG_LEVEL=DEBUG
ROUTER_LOG_FORMAT=json  # easier to parse programmatically
```

View logs with request tracking:

```bash
# Make request with tracking ID
curl -H "X-Request-ID: debug-123" http://localhost:11436/v1/models

# Follow logs
docker logs -f smarterrouter | grep "debug-123"

# Or for JSON logs, use jq
docker logs smarterrouter 2>&1 | jq 'select(.request_id=="debug-123")'
```

Check detailed request logs (if enabled with `--enable-request-logging` flag):
```
logs/detailed_logs/<timestamp>/
├── request.json          # Full request payload
├── response.json         # Full response or error
├── streaming_chunks.jsonl # SSE chunks if streaming
└── metadata.json         # Timing, model selected, scores
```

---

## Still Stuck?

1. **Check the logs** - Most issues are clearly explained there
2. **Run health check** - `curl http://localhost:11436/health`
3. **Verify backend connectivity** - `curl http://localhost:11434/api/tags`
4. **Open an issue** - Include:
   - SmarterRouter version (from git commit or image tag)
   - Full error messages
   - Relevant log snippets
   - Steps to reproduce
   - Your configuration (redact API keys!)
