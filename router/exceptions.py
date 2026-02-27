"""
Exception hierarchy for SmarterRouter.

Provides consistent exception types across the codebase.
"""


class RouterError(Exception):
    """Base exception for all router-related errors."""

    pass


class RouterConfigError(RouterError):
    """Configuration error (invalid settings, missing required config)."""

    pass


class RouterDatabaseError(RouterError):
    """Database operation error (connection, query, transaction failures)."""

    pass


class RouterBackendError(RouterError):
    """LLM backend error (model unavailable, generation failure, timeout)."""

    pass


class RouterProfilingError(RouterError):
    """Model profiling error (profiling failure, invalid results)."""

    pass


class RouterVRAMError(RouterError):
    """VRAM management error (insufficient VRAM, allocation failure)."""

    pass


class RouterSecurityError(RouterError):
    """Security violation (invalid input, authorization failure, path traversal)."""

    pass


class RouterValidationError(RouterError):
    """Validation error (invalid request, malformed data)."""

    pass


class RouterRateLimitError(RouterError):
    """Rate limiting error (too many requests)."""

    pass


# Compatibility aliases for existing exceptions
ProviderDBError = RouterDatabaseError
VRAMExceededError = RouterVRAMError
