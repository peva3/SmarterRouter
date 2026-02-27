#!/bin/bash
# One-line Docker deployment script for SmarterRouter
# Auto-detects GPU hardware and runs container with optimal configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
IMAGE_NAME="smarterrouter:latest"
CONTAINER_NAME="smarterrouter"
PORT="11436"
DATA_DIR="./data"
ENV_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "One-line Docker deployment for SmarterRouter"
            echo ""
            echo "Options:"
            echo "  --image IMAGE       Docker image name (default: smarterrouter:latest)"
            echo "  --name NAME         Container name (default: smarterrouter)"
            echo "  --port PORT         Host port to map (default: 11436)"
            echo "  --data-dir DIR      Directory for persistent data (default: ./data)"
            echo "  --env-file FILE     Path to .env file (default: auto-generated)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 SmarterRouter One-Line Deployment"
echo "====================================="

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"
echo "📁 Data directory: $DATA_DIR"

# GPU detection
echo "🔍 Detecting GPU hardware..."

GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    GPU_FLAGS="--gpus all"
elif [[ -d "/dev/dri" ]]; then
    if [[ -e "/dev/kfd" ]]; then
        echo "✅ AMD GPU detected (ROCm)"
        GPU_FLAGS="--device /dev/kfd --device /dev/dri"
    else
        echo "✅ Intel GPU detected"
        GPU_FLAGS="--device /dev/dri"
    fi
else
    echo "ℹ️  No GPU detected, running in CPU-only mode"
fi

# Build Docker run command
CMD="docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p $PORT:11436 \
    -v $(realpath "$DATA_DIR"):/app/data:rw"

# Add GPU flags if any
if [[ -n "$GPU_FLAGS" ]]; then
    CMD="$CMD $GPU_FLAGS"
fi

# Add environment file if specified
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    echo "📝 Using environment file: $ENV_FILE"
    CMD="$CMD --env-file $ENV_FILE"
else
    echo "📝 Using auto-generated configuration"
fi

# Add image
CMD="$CMD $IMAGE_NAME"

echo ""
echo "📦 Running container with command:"
echo "$CMD"
echo ""

# Run the container
eval $CMD

echo "✅ Container '$CONTAINER_NAME' started successfully!"
echo ""
echo "📊 Next steps:"
echo "   1. Check logs: docker logs -f $CONTAINER_NAME"
echo "   2. View health: docker exec $CONTAINER_NAME python -m router.cli check"
echo "   3. Access router: http://localhost:$PORT"
echo "   4. Stop container: docker stop $CONTAINER_NAME"
echo "   5. Remove container: docker rm $CONTAINER_NAME"
echo ""
echo "🔧 For production deployment, use docker-compose.yml with GPU configuration."