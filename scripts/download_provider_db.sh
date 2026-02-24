#!/bin/bash
# Download latest provider.db from GitHub
# Run this script via cron: 0 1 * * * /app/hubrouter/scripts/download_provider_db.sh

set -e

PROVIDER_DB_URL="https://raw.githubusercontent.com/peva3/smarterrouter-provider/refs/heads/main/data/provider.db"
DATA_DIR="/app/hubrouter/data"
PROVIDER_DB="${DATA_DIR}/provider.db"
LOG_FILE="${DATA_DIR}/provider_db_download.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting provider.db download" >> "$LOG_FILE"

# Check if wget is available
if ! command -v wget &> /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: wget not found" >> "$LOG_FILE"
    exit 1
fi

# Create temp file
TEMP_DB="${PROVIDER_DB}.tmp.$$"

# Download with timeout and retry
if wget --timeout=60 --tries=3 -O "$TEMP_DB" "$PROVIDER_DB_URL" 2>> "$LOG_FILE"; then
    # Verify it's a valid SQLite database
    if python3 -c "import sqlite3; conn = sqlite3.connect('$TEMP_DB'); conn.close()" 2>> "$LOG_FILE"; then
        # Atomic replace
        mv "$TEMP_DB" "$PROVIDER_DB"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: provider.db updated" >> "$LOG_FILE"
        
        # Get model count
        MODEL_COUNT=$(python3 -c "import sqlite3; conn = sqlite3.connect('$PROVIDER_DB'); cur = conn.execute('SELECT COUNT(*) FROM model_benchmarks'); print(cur.fetchone()[0])" 2>> "$LOG_FILE")
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: $MODEL_COUNT models in database" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Downloaded file is not a valid SQLite database" >> "$LOG_FILE"
        rm -f "$TEMP_DB"
        exit 1
    fi
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Download failed" >> "$LOG_FILE"
    rm -f "$TEMP_DB"
    exit 1
fi
