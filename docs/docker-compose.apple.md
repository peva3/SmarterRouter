# Apple Silicon (M1/M2/M3) - Native Installation Guide
#
# IMPORTANT: Docker Desktop on macOS CANNOT pass GPU to Linux containers.
# You must run SmarterRouter natively on the macOS host for GPU support.
#
# Apple Silicon uses unified memory where CPU and GPU share system RAM.
# SmarterRouter estimates GPU memory as 75% of total RAM by default.

## Native Installation

### 1. Prerequisites
```bash
# Check your macOS version and chip
system_profiler SPHardwareDataType

# Expected output should show:
# Chip: Apple M1/M2/M3/M4
# Memory: 8/16/32/64/128 GB
```

### 2. Install Dependencies
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+
brew install python@3.11

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install SmarterRouter
```bash
# Clone repository
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy template
cp ENV_DEFAULT .env

# Edit configuration
nano .env
```

**Recommended settings for Apple Silicon:**
```env
# Backend URL (if running Ollama on same machine)
ROUTER_OLLAMA_URL=http://localhost:11434

# Apple Silicon specific (optional, auto-detected if not set)
ROUTER_APPLE_UNIFIED_MEMORY_GB=16  # Set to your Mac's RAM

# VRAM monitoring
ROUTER_VRAM_MONITOR_ENABLED=true
```

### 5. Run SmarterRouter
```bash
# Development
python -m uvicorn main:app --host 0.0.0.0 --port 11436

# Production (with gunicorn)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:11436
```

### 6. Verify Installation
```bash
# Check health
curl http://localhost:11436/health

# Check VRAM (should show unified memory)
curl http://localhost:11436/admin/vram
```

## Running as a LaunchAgent (Auto-start on boot)

Create `~/Library/LaunchAgents/com.smarterrouter.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.smarterrouter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/SmarterRouter/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>11436</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/SmarterRouter</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/smarterrouter.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/smarterrouter.error.log</string>
</dict>
</plist>
```

Load the service:
```bash
launchctl load ~/Library/LaunchAgents/com.smarterrouter.plist
```

## Memory Configuration

Apple Silicon GPU memory is estimated based on unified memory:

| Mac RAM | Estimated GPU Memory (75%) | Configurable |
|---------|---------------------------|--------------|
| 8 GB    | 6 GB                      | Yes |
| 16 GB   | 12 GB                     | Yes |
| 32 GB   | 24 GB                     | Yes |
| 64 GB   | 48 GB                     | Yes |
| 128 GB  | 96 GB                     | Yes |

Override with `ROUTER_APPLE_UNIFIED_MEMORY_GB` if auto-detection fails.

## Troubleshooting

### "Apple Silicon not detected"
- Ensure you're running on macOS with ARM64: `uname -m` should show `arm64`
- Check system profiler: `system_profiler SPHardwareDataType`

### "Memory estimate is wrong"
- Manually set in `.env`: `ROUTER_APPLE_UNIFIED_MEMORY_GB=16`

### "Running in Docker shows no GPU"
- Docker Desktop on macOS cannot pass GPU to containers
- Run natively on host instead (see instructions above)
