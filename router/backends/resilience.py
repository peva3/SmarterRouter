from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from router.circuit_breaker import CircuitBreakerConfig, get_circuit_breaker
from router.backends.retry import QuotaExhaustedError, retry_operation

T = TypeVar("T")


async def with_backend_resilience(
    operation_name: str,
    operation: Callable[[], Awaitable[T]],
    config: Any,
) -> T:
    """Apply circuit breaker then retry policy for backend operations.

    Distinguishes quota-exhausted failures from transient 429s:
    - Quota failures get a longer circuit breaker timeout and skip retry.
    - Transient failures use the standard retry + circuit breaker cycle.

    Order matters:
    - Circuit breaker wraps the retried operation so one logical operation
      counts as one success/failure in breaker state.
    - Retry still handles transient errors inside the breaker execution.
    """

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
            quota_reset_timeout=getattr(
                config, "backend_circuit_breaker_quota_reset_timeout", 3600.0
            ),
        ),
    )

    if not breaker.is_call_allowed():
        from router.circuit_breaker import CircuitBreakerOpenError

        raise CircuitBreakerOpenError(breaker.name, "Circuit breaker is open")

    try:
        result = await with_retry()
        await breaker.record_success()
        return result
    except QuotaExhaustedError as e:
        await breaker.record_failure(failure_type="quota")
        raise
    except Exception:
        await breaker.record_failure()
        raise
