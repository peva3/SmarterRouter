"""API route modules for SmarterRouter."""

from router.api.admin import router as admin_router
from router.api.chat import router as chat_router
from router.api.health import router as health_router
from router.api.models import router as models_router

__all__ = ["admin_router", "chat_router", "health_router", "models_router"]
