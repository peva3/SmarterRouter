"""Retry utilities for robust backend HTTP calls.

Distinguishes transient rate-limit errors (should retry with backoff)
from quota-exhausted errors (should NOT retry — wait for period reset).
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

# Keywords in 429 response bodies that indicate quota exhaustion
# rather than a transient rate-limit that will resolve with backoff.
QUOTA_KEYWORDS = [
    # Billing/quota period signals (longer-than-minute windows)
    # These indicate exhaustion of a quota/cap rather than a transient rate-limit.
    # Generic "rate limit exceeded" or "too many requests" are NOT included here.
    "quota exceeded",
    "quota exhausted",
    "out of credits",
    "insufficient quota",
    "monthly quota",
    "monthly limit",
    "daily quota",
    "daily limit",
    "billing limit",
    "cap reached",
    "plan limit",
    "tier limit",
    "credit limit",
]


class QuotaExhaustedError(Exception):
    """Raised when a 429 response indicates quota exhaustion
    (not a transient rate-limit). Retrying would be futile until
    the quota period resets."""

    def __init__(self, message: str, response: httpx.Response | None = None):
        super().__init__(message)
        self.response = response


def is_quota_exhausted(e: Exception) -> bool:
    """Check if an HTTPStatusError 429 response indicates quota exhaustion
    by inspecting the response body for known quota/cap keywords."""
    if not isinstance(e, httpx.HTTPStatusError):
        return False
    if e.response.status_code != 429:
        return False
    try:
        body = e.response.text.lower()
    except Exception:
        return False
    return any(kw in body for kw in QUOTA_KEYWORDS)


def is_retryable_exception(e: Exception) -> bool:
    """Check if an exception is retryable (timeout or transient HTTP error).

    Returns False for quota-exhausted 429s — those are promoted to
    QuotaExhaustedError by the caller instead.
    """
    if isinstance(e, httpx.TimeoutException):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        if is_quota_exhausted(e):
            return False
        return e.response.status_code in (429, 500, 502, 503, 504)
    if isinstance(e, httpx.RequestError):
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
        QuotaExhaustedError: If a 429 response indicates quota/cap exhaustion.
        The last exception if all retries fail.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError) and is_quota_exhausted(e):
                body = e.response.text[:300]
                raise QuotaExhaustedError(
                    f"Quota exhausted (429): {body}", response=e.response
                ) from e
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
