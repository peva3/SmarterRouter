"""SmarterRouter - AI-powered LLM router.

Main entry point that creates the FastAPI application and wires together
all components: middleware, API routers, lifecycle events, and shared state.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router.api import admin_router, chat_router, health_router, models_router
from router.config import settings
from router.lifecycle import lifespan
from router.middleware import register_middleware

# Re-export shared state and utilities for test compatibility
from router.state import (
    app_state,
    get_settings,
    verify_admin_token,
    _get_client_ip,
    _ip_in_whitelist,
)

logger = logging.getLogger(__name__)

# Create FastAPI app with lifespan management
app = FastAPI(
    title="SmarterRouter",
    description="AI-powered LLM router that intelligently selects the best model",
    version="2.2.0",
    lifespan=lifespan,
)

# CORS middleware (Item #22)
# Parse comma-separated origins
cors_origins_list = [origin.strip() for origin in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=[method.strip() for method in settings.cors_allow_methods.split(",")],
    allow_headers=[header.strip() for header in settings.cors_allow_headers.split(",")],
    max_age=settings.cors_max_age,
)
logger.info(f"CORS configured with origins: {cors_origins_list}")

# Optional gzip compression
# Note: Import inside conditional to avoid hard dependency
# This allows the app to start even if starlette is not installed
# (though it should be in requirements.txt)
try:
    from starlette.middleware.gzip import GZipMiddleware

    from router.config import settings

    if settings.enable_response_compression:
        app.add_middleware(
            GZipMiddleware, minimum_size=settings.compression_minimum_size
        )
        logger.info("GZip compression enabled")
except ImportError:
    logger.warning("Starlette not available, GZip compression disabled")

# Register all middleware
register_middleware(app)

# Include all API routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(admin_router)

logger.info("SmarterRouter application initialized")


# Entry points for running the server
async def main_async() -> None:
    """Async entry point for running the server programmatically."""
    import uvicorn

    from router.config import settings

    config = uvicorn.Config(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


def main() -> None:
    """Synchronous entry point for running from command line."""
    import asyncio

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
