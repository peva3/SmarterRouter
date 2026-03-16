"""Retry utilities for robust backend HTTP calls."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

T = TypeVar("T")


def is_retryable_exception(e: Exception) -> bool:
    """Check if an exception is retryable (timeout or transient HTTP error)."""
    if isinstance(e, httpx.TimeoutException):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503, 504)
    if isinstance(e, httpx.RequestError):
        # Network errors (connection, etc.) are retryable
        return True
    return False


async def retry_operation(
    operation: Callable[[], Awaitable[T]],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    """Retry an async operation with exponential backoff.

    Args:
        operation: Async callable to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (will be multiplied by 2^attempt)
        max_delay: Maximum delay between retries

    Returns:
        Result of the operation

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            if not is_retryable_exception(e):
                raise
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                await asyncio.sleep(delay)
            else:
                break
    if last_exception:
        raise last_exception
    raise RuntimeError("retry_operation completed without returning or raising")
