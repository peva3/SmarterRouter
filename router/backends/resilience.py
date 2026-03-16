from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from router.circuit_breaker import CircuitBreakerConfig, get_circuit_breaker

T = TypeVar("T")


async def with_backend_resilience(
    operation_name: str,
    operation: Callable[[], Awaitable[T]],
    config: Any,
) -> T:
    """Apply circuit breaker then retry policy for backend operations.

    Order matters:
    - Circuit breaker wraps the retried operation so one logical operation
      counts as one success/failure in breaker state.
    - Retry still handles transient errors inside the breaker execution.
    """
    from router.backends.retry import retry_operation

    async def with_retry() -> T:
        if config.backend_retry_enabled:
            return await retry_operation(
                operation,
                max_retries=config.backend_max_retries,
                base_delay=config.backend_retry_base_delay,
                max_delay=config.backend_retry_max_delay,
            )
        return await operation()

    if not config.backend_circuit_breaker_enabled:
        return await with_retry()

    breaker = await get_circuit_breaker(
        operation_name,
        CircuitBreakerConfig(
            failure_threshold=config.backend_circuit_breaker_failure_threshold,
            reset_timeout=config.backend_circuit_breaker_reset_timeout,
            half_open_max_attempts=config.backend_circuit_breaker_half_open_max_attempts,
            sliding_window_size=config.backend_circuit_breaker_sliding_window_size,
        ),
    )

    return await breaker.execute(with_retry)
