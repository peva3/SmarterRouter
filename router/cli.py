"""
SmarterRouter Command Line Interface.

Provides interactive setup, configuration validation, and management tools.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from router.config import Settings
from router.gpu_backends import GPUBackendManager

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )


async def detect_ollama(urls: list[str] | None = None) -> tuple[str | None, list[str]]:
    """
    Detect Ollama installation by trying multiple endpoints.

    Returns:
        Tuple of (detected_url, list_of_available_models)
    """
    if urls is None:
        urls = [
            "http://localhost:11434",
            "http://172.17.0.1:11434",  # Docker host IP
            "http://host.docker.internal:11434",  # Docker Desktop
            "http://ollama:11434",  # Docker service name
        ]

    timeout = httpx.Timeout(5.0, connect=3.0)

    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Try /api/tags endpoint
                response = await client.get(f"{url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    logger.debug(f"✓ Ollama detected at {url} with {len(models)} models")
                    return url, models
        except Exception as e:
            logger.debug(f"  Ollama not found at {url}: {e}")
            continue

    return None, []


async def detect_hardware() -> dict[str, Any]:
    """
    Detect GPU hardware and available VRAM.

    Returns:
        Dictionary with hardware information
    """
    hardware: dict[str, Any] = {
        "gpu_vendor": None,
        "total_vram_gb": 0.0,
        "gpu_count": 0,
        "detected": False,
    }

    try:
        gpu_manager = GPUBackendManager()

        if gpu_manager.backends:
            hardware["gpu_vendor"] = gpu_manager.backends[0].vendor
            hardware["gpu_count"] = gpu_manager.gpu_count

            # Get total VRAM across all GPUs
            total_vram = gpu_manager.get_total_vram()

            hardware["total_vram_gb"] = total_vram
            hardware["detected"] = True

            logger.debug(
                f"✓ GPU detected: {hardware['gpu_vendor']} "
                f"({hardware['gpu_count']} GPU(s), {total_vram:.1f}GB VRAM)"
            )
        else:
            logger.debug("  No GPU detected, running in CPU-only mode")

    except Exception as e:
        logger.debug(f"  GPU detection failed: {e}")

    return hardware


async def suggest_settings(
    ollama_url: str | None, models: list[str], hardware: dict
) -> dict[str, Any]:
    """
    Suggest optimal settings based on detected environment.

    Returns:
        Dictionary with suggested configuration
    """
    suggestions = {
        "ollama_url": ollama_url or "http://localhost:11434",
        "pinned_model": None,
        "quality_preference": 0.5,  # balanced
        "vram_monitor_enabled": hardware["detected"],
        "vram_max_total_gb": None,
        "provider": "ollama",
    }

    # Suggest pinned model (smallest available model)
    if models:
        # Sort by estimated size (crude heuristic)
        def estimate_size(name: str) -> int:
            name_lower = name.lower()
            if "tiny" in name_lower or "mini" in name_lower:
                return 1
            elif "3b" in name_lower or "4b" in name_lower:
                return 4
            elif "7b" in name_lower or "8b" in name_lower:
                return 8
            elif "13b" in name_lower or "14b" in name_lower:
                return 14
            elif "34b" in name_lower or "40b" in name_lower:
                return 40
            elif "70b" in name_lower:
                return 70
            else:
                return 20  # default medium size

        sorted_models = sorted(models, key=estimate_size)
        suggestions["pinned_model"] = sorted_models[0]

        # Adjust quality preference based on hardware
        if hardware["detected"]:
            if hardware["total_vram_gb"] >= 24:
                suggestions["quality_preference"] = 0.8  # favor quality
            elif hardware["total_vram_gb"] >= 8:
                suggestions["quality_preference"] = 0.6  # slightly quality
            else:
                suggestions["quality_preference"] = 0.4  # favor speed
        else:
            suggestions["quality_preference"] = 0.3  # CPU-only: favor speed

    # Suggest VRAM max (90% of detected VRAM)
    if hardware["detected"] and hardware["total_vram_gb"] > 0:
        suggestions["vram_max_total_gb"] = round(hardware["total_vram_gb"] * 0.9, 1)

    return suggestions


def generate_env_file(suggestions: dict[str, Any], output_path: Path) -> None:
    """
    Generate .env file with suggested settings and explanations.
    """
    env_content = """# SmarterRouter Configuration - Generated by setup wizard
# Created: {timestamp}

# =============================================================================
# BACKEND PROVIDER CONFIGURATION
# =============================================================================

# Provider selection: ollama, llama.cpp, or openai (default: ollama)
ROUTER_PROVIDER={provider}

# === Ollama Backend Settings ===
# URL of your Ollama instance
ROUTER_OLLAMA_URL={ollama_url}

# Model to keep permanently loaded in VRAM (optional)
# Useful for keeping a small/fast model ready for simple queries
ROUTER_PINNED_MODEL={pinned_model}

# Quality vs Speed tradeoff (0.0 = fastest, 1.0 = highest quality)
ROUTER_QUALITY_PREFERENCE={quality_preference}

# =============================================================================
# VRAM MONITORING
# =============================================================================

# Enable VRAM monitoring (auto-detects NVIDIA, AMD, Intel, Apple Silicon)
ROUTER_VRAM_MONITOR_ENABLED={vram_monitor_enabled}

# Maximum VRAM budget for models (GB)
# Set to 90% of your total VRAM for safety margin
ROUTER_VRAM_MAX_TOTAL_GB={vram_max_total_gb}

# =============================================================================
# ADMIN SECURITY
# =============================================================================

# **REQUIRED FOR PRODUCTION** - Set a secure API key to protect admin endpoints
# ROUTER_ADMIN_API_KEY=your-secret-key-here

# =============================================================================
# ADVANCED SETTINGS (Optional)
# =============================================================================

# Model filtering (comma-separated glob patterns, case-insensitive)
# ROUTER_MODEL_FILTER_INCLUDE=
# ROUTER_MODEL_FILTER_EXCLUDE=

# External provider support (OpenAI, Anthropic, Google, etc.)
# ROUTER_EXTERNAL_PROVIDERS_ENABLED=false
# ROUTER_EXTERNAL_PROVIDERS=openai,anthropic

# Embeddings configuration
# ROUTER_EMBEDDINGS_ENABLED=true
# ROUTER_EMBEDDINGS_MODEL=nomic-embed-text

""".format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        provider=suggestions["provider"],
        ollama_url=suggestions["ollama_url"],
        pinned_model=suggestions["pinned_model"] or "",
        quality_preference=suggestions["quality_preference"],
        vram_monitor_enabled=str(suggestions["vram_monitor_enabled"]).lower(),
        vram_max_total_gb=suggestions["vram_max_total_gb"] or "",
    )

    output_path.write_text(env_content)
    logger.info(f"✓ Generated configuration file: {output_path}")


async def check_configuration(config_path: Path | None = None) -> dict[str, Any]:
    """
    Validate current configuration and connections.

    Returns:
        Dictionary with validation results
    """
    results: dict[str, Any] = {
        "config_file_exists": False,
        "config_valid": False,
        "ollama_reachable": False,
        "models_available": False,
        "gpu_detected": False,
        "issues": [],
    }

    # Check config file
    if config_path is None:
        config_path = Path(".env")

    if config_path.exists():
        results["config_file_exists"] = True
        try:
            # Try to load settings
            settings = Settings(_env_file=str(config_path))  # type: ignore[call-arg]
            results["config_valid"] = True
            ollama_url = settings.ollama_url
        except Exception as e:
            results["issues"].append(f"Config validation failed: {e}")
            ollama_url = "http://localhost:11434"
    else:
        results["issues"].append("Configuration file (.env) not found")
        ollama_url = "http://localhost:11434"

    # Check Ollama connection
    ollama_detected, models = await detect_ollama([ollama_url])
    if ollama_detected:
        results["ollama_reachable"] = True
        results["models_available"] = len(models) > 0
        if not results["models_available"]:
            results["issues"].append("Ollama reachable but no models found")
    else:
        results["issues"].append(f"Ollama not reachable at {ollama_url}")

    # Check GPU
    hardware = await detect_hardware()
    results["gpu_detected"] = hardware["detected"]

    return results


def print_check_results(results: dict[str, Any]) -> None:
    """Print configuration check results in a user-friendly format."""
    print("\n" + "=" * 60)
    print("CONFIGURATION CHECK RESULTS")
    print("=" * 60)

    # Status indicators
    status_icon = {True: "✓", False: "✗"}

    print(f"\nConfiguration File: {status_icon[results['config_file_exists']]}", end="")
    if results["config_file_exists"]:
        print(" (.env found)")
    else:
        print(" (missing)")

    if results["config_valid"]:
        print("Config Validation: ✓ (valid)")
    else:
        print("Config Validation: ✗ (invalid)")

    print(f"Ollama Connection: {status_icon[results['ollama_reachable']]}", end="")
    if results["ollama_reachable"]:
        print(" (reachable)")
    else:
        print(" (unreachable)")

    print(f"Models Available: {status_icon[results['models_available']]}", end="")
    if results["models_available"]:
        print(" (yes)")
    else:
        print(" (none)")

    print(f"GPU Detected: {status_icon[results['gpu_detected']]}", end="")
    if results["gpu_detected"]:
        print(" (yes)")
    else:
        print(" (CPU-only mode)")

    if results["issues"]:
        print(f"\n⚠️  Issues Found ({len(results['issues'])}):")
        for issue in results["issues"]:
            print(f"   • {issue}")

    print("\n" + "=" * 60)


async def setup_interactive() -> None:
    """
    Interactive setup wizard for SmarterRouter.
    """
    print("\n" + "=" * 60)
    print("SMARTERROUTER SETUP WIZARD")
    print("=" * 60)

    print("\n🔍 Detecting Ollama installation...")
    ollama_url, models = await detect_ollama()

    if ollama_url:
        print(f"✓ Found Ollama at {ollama_url}")
        if models:
            print(
                f"📊 Found {len(models)} models: {', '.join(models[:5])}"
                + (f" and {len(models) - 5} more..." if len(models) > 5 else "")
            )
        else:
            print("⚠️  No models found in Ollama. Please pull some models first.")
            print("   Example: `ollama pull llama3.2`")
    else:
        print("✗ Ollama not detected. Please install Ollama first.")
        print("  Download from: https://ollama.com")
        print("\nContinue with default settings? [Y/n]: ", end="")
        response = input().strip().lower()
        if response not in ["", "y", "yes"]:
            print("Setup cancelled.")
            return
        ollama_url = "http://localhost:11434"

    print("\n⚙️  Detecting hardware...")
    hardware = await detect_hardware()

    if hardware["detected"]:
        print(f"✓ {hardware['gpu_vendor'].upper()} GPU detected")
        print(f"  {hardware['gpu_count']} GPU(s), {hardware['total_vram_gb']:.1f}GB VRAM total")
    else:
        print("  No GPU detected, running in CPU-only mode")

    print("\n💡 Suggested settings:")
    suggestions = await suggest_settings(ollama_url, models, hardware)

    print(f"  • Ollama URL: {suggestions['ollama_url']}")
    if suggestions["pinned_model"]:
        print(f"  • Pinned model: {suggestions['pinned_model']} (smallest available)")
    print(
        f"  • Quality preference: {suggestions['quality_preference']:.1f} "
        + f"({'speed' if suggestions['quality_preference'] < 0.5 else 'quality' if suggestions['quality_preference'] > 0.5 else 'balanced'})"
    )
    print(
        f"  • VRAM monitoring: {'enabled' if suggestions['vram_monitor_enabled'] else 'disabled'}"
    )
    if suggestions["vram_max_total_gb"]:
        print(
            f"  • VRAM budget: {suggestions['vram_max_total_gb']:.1f}GB "
            + f"({hardware['total_vram_gb']:.1f}GB total × 90%)"
        )

    print("\n📝 Generate .env file? [Y/n]: ", end="")
    response = input().strip().lower()

    if response in ["", "y", "yes"]:
        output_path = Path(".env")
        if output_path.exists():
            print(f"\n⚠️  File {output_path} already exists.")
            print("Overwrite? [y/N]: ", end="")
            overwrite = input().strip().lower()
            if overwrite not in ["y", "yes"]:
                print("Setup completed without overwriting .env")
                return

        generate_env_file(suggestions, output_path)
        print("\n✅ Setup completed!")
        print("\nNext steps:")
        print("1. Review the generated .env file")
        print("2. Start SmarterRouter: docker-compose up -d")
        print("3. Check logs: docker-compose logs -f smarterrouter")
        print("4. Access the router at: http://localhost:11436")
    else:
        print("\nSetup completed without generating .env file")

    print("\n" + "=" * 60)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmarterRouter CLI - Intelligent LLM router management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup          Interactive setup wizard
  %(prog)s check          Validate configuration
  %(prog)s generate-env   Generate .env file with defaults
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", title="commands", help="available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard")
    setup_parser.add_argument(
        "--non-interactive", action="store_true", help="Generate .env without prompts"
    )

    # Check command
    check_parser = subparsers.add_parser("check", help="Validate configuration and connections")
    check_parser.add_argument(
        "--config",
        type=Path,
        default=Path(".env"),
        help="Path to configuration file (default: .env)",
    )

    # Generate-env command
    gen_parser = subparsers.add_parser(
        "generate-env", help="Generate .env file with default settings"
    )
    gen_parser.add_argument(
        "--output", "-o", type=Path, default=Path(".env"), help="Output file path (default: .env)"
    )
    gen_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing file without prompting"
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "setup":
        asyncio.run(setup_interactive())

    elif args.command == "check":
        results = asyncio.run(check_configuration(args.config))
        print_check_results(results)

        # Exit with error code if there are critical issues
        if not results["ollama_reachable"]:
            sys.exit(1)

    elif args.command == "generate-env":

        async def generate_env():
            hardware = await detect_hardware()
            suggestions = await suggest_settings(None, [], hardware)

            if args.output.exists() and not args.overwrite:
                logger.error(f"File {args.output} already exists. Use --overwrite to replace.")
                sys.exit(1)

            generate_env_file(suggestions, args.output)

        asyncio.run(generate_env())

    else:
        # No command specified, show help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
