#!/usr/bin/env bash
# Test script to verify VRAM measurement is working correctly

set -e

OLLAMA_URL="${ROUTER_OLLAMA_URL:-http://172.17.0.1:11435}"
ROUTER_URL="${ROUTER_URL:-http://localhost:11436}"

echo "=== VRAM Measurement Test ==="
echo "Ollama URL: $OLLAMA_URL"
echo "Router URL: $ROUTER_URL"
echo

echo "1. Testing Ollama /api/ps endpoint (should show running models with VRAM):"
curl -s "$OLLAMA_URL/api/ps" | python3 -m json.tool 2>/dev/null || echo "  Failed to reach Ollama"
echo

echo "2. Testing nvidia-smi directly:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo

echo "3. Running a quick chat to load a model:"
echo "   Loading llama3.2:1b..."
curl -s -X POST "$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}], "model": "llama3.2:1b"}' > /dev/null
echo "   Done."
echo

echo "4. Checking if model is now in Ollama /api/ps:"
curl -s "$OLLAMA_URL/api/ps" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if models:
    print('  Running models:')
    for m in models:
        name = m.get('name', 'unknown')
        vram = m.get('size_vram', 0) / (1024**3)  # Convert to GB
        size = m.get('size', 0) / (1024**3)
        print(f'    - {name}: {vram:.2f}GB VRAM (total: {size:.2f}GB)')
else:
    print('  No models currently running')
"
echo

echo "5. Checking router VRAM status:"
curl -s "$ROUTER_URL/admin/vram" | python3 -m json.tool 2>/dev/null || echo "  Admin endpoint not available"
echo

echo "=== Test Complete ==="
echo
echo "Expected behavior:"
echo "- /api/ps should show llama3.2:1b with VRAM usage (e.g., 1-2GB)"
echo "- nvidia-smi should show GPU memory being used"
echo "- /admin/vram should reflect the usage"
