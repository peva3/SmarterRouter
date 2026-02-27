#!/usr/bin/env python3
"""
Docker entrypoint for SmarterRouter.

This script handles:
1. Configuration setup on first run
2. Environment validation
3. Starting the FastAPI server
"""

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

# Add parent directory to path so we can import router modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.cli import (
    check_configuration,
    detect_hardware,
    detect_ollama,
    generate_env_file,
    suggest_settings,
)

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for entrypoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def auto_setup():
    """
    Automatic setup for Docker containers.

    Checks for existing configuration and creates optimal .env if missing.
    Returns True if setup was performed, False if using existing config.
    """
    # First check if we can load settings from environment variables or .env file
    try:
        from router.config import Settings

        # Try to load settings - this will use .env file if exists, then environment variables
        settings = Settings()
        logger.info("Configuration loaded successfully from environment/.env")
        # Check if we have essential settings
        if settings.ollama_url and settings.provider:
            logger.info(
                f"Using existing configuration (provider: {settings.provider}, ollama: {settings.ollama_url})"
            )
            return False
    except Exception as e:
        logger.debug(f"Could not load settings: {e}")
        # Continue to check for .env file

    env_path = Path("/app/data/.env")

    # Also check current directory for backward compatibility
    if not env_path.exists():
        env_path = Path(".env")

    if env_path.exists():
        logger.info(f"Using existing configuration file: {env_path}")
        return False

    logger.info("No configuration found, running automatic setup...")

    # Detect environment
    ollama_url, models = await detect_ollama()
    hardware = await detect_hardware()

    if not ollama_url:
        logger.warning("Ollama not detected. Using default configuration.")
        ollama_url = "http://host.docker.internal:11434"

    # Generate suggested settings
    suggestions = await suggest_settings(ollama_url, models, hardware)

    # Override defaults for Docker environment
    suggestions["ollama_url"] = ollama_url

    # Ensure data directory exists
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate .env file
    generate_env_file(suggestions, env_path)

    logger.info(f"Generated configuration at {env_path}")
    logger.info("Next: Review the configuration and adjust as needed.")

    return True


async def validate_environment():
    """Validate configuration and connections."""
    import os
    from pathlib import Path

    logger.info("Validating environment...")

    # Check for .env in data directory first
    env_path = Path("/app/data/.env")
    if not env_path.exists():
        env_path = Path(".env")

    results = await check_configuration(env_path if env_path.exists() else None)

    # Log results
    if results["config_file_exists"]:
        logger.info("✓ Configuration file found")
    else:
        logger.warning("No configuration file found")

    if results["config_valid"]:
        logger.info("✓ Configuration is valid")
    else:
        logger.error("Configuration validation failed")

    if results["ollama_reachable"]:
        logger.info("✓ Ollama is reachable")
    else:
        logger.error("Ollama is not reachable")

    if results["models_available"]:
        logger.info("✓ Models are available")
    else:
        logger.warning("No models available")

    if results["gpu_detected"]:
        logger.info("✓ GPU detected")
    else:
        logger.info("Running in CPU-only mode")

    if results["issues"]:
        logger.warning(f"Found {len(results['issues'])} issue(s):")
        for issue in results["issues"]:
            logger.warning(f"  • {issue}")

    # Check database directory permissions
    try:
        from router.config import settings

        if "sqlite" in settings.database_url.lower():
            # Extract file path
            db_path = settings.database_url.replace("sqlite:///", "").replace("sqlite://", "")
            if "?" in db_path:
                db_path = db_path.split("?")[0]
            db_file = Path(db_path)
            db_dir = db_file.parent
            if db_dir.exists():
                if not os.access(db_dir, os.W_OK):
                    logger.error(
                        f"Database directory {db_dir} is not writable by current user. This will cause migration failures."
                    )
                    logger.error(
                        f"Ensure the directory has correct permissions (user {os.getuid()})."
                    )
                    # Non-fatal, but will likely cause errors later
            else:
                logger.info(f"Database directory {db_dir} does not exist, will be created")
    except Exception as e:
        logger.debug(f"Could not check database permissions: {e}")

    return results


async def start_server():
    """Start the FastAPI server programmatically in the existing event loop."""
    from router.config import settings

    logger.info(f"Starting SmarterRouter on {settings.host}:{settings.port}")

    config = uvicorn.Config(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Main entrypoint function."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("SmarterRouter Docker Entrypoint")
    logger.info("=" * 60)

    # Run auto-setup if no config exists
    await auto_setup()

    # Validate environment
    validation_results = await validate_environment()

    # Warn about critical issues but continue
    if not validation_results["ollama_reachable"]:
        logger.error("CRITICAL: Ollama is not reachable. Router may not function correctly.")
        logger.error("Ensure Ollama is running and accessible from the container.")

    if not validation_results["models_available"]:
        logger.warning("No models available. Please pull models in Ollama.")
        logger.warning("Example: docker exec -it ollama ollama pull llama3.2")

    # Start server
    logger.info("Starting server...")
    await start_server()


if __name__ == "__main__":
    asyncio.run(main())
