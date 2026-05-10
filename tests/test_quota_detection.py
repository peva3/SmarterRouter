"""Tests for quota-exhausted vs rate-limit disambiguation in circuit breakers.

Verifies that 429 responses with quota/cap keywords in the body
are treated as non-retryable quota exhaustion, while 429s without
those keywords are treated as transient rate-limits.
"""

import time

import httpx
import pytest

from router.backends.retry import (
    QUOTA_KEYWORDS,
    QuotaExhaustedError,
    is_quota_exhausted,
    is_retryable_exception,
    retry_operation,
)
from router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


# ---------------------------------------------------------------------------
# 1. Quota keyword detection
# ---------------------------------------------------------------------------

class TestIsQuotaExhausted:
    """Verify 429 response body keyword matching."""

    QUOTA_BODIES = [
        "Monthly quota exceeded. Resets at 2026-06-01",
        "API quota exhausted. Out of credits.",
        "Daily limit reached. Please try again tomorrow.",
        "Insufficient quota to process this request.",
        "Billing limit reached. Upgrade your plan.",
        "Cap reached for this billing period.",
        "Plan limit exceeded for your account tier.",
        "Credit limit reached for this month.",
    ]

    TRANSIENT_BODIES = [
        "Rate limit: 10 requests per minute. Retry after 60 seconds.",
        "Too Many Requests. Slow down.",
        "Please wait before sending more requests.",
        "",
        '{"error": {"message": "rate limit"}}',
        "Server busy, try again shortly.",
    ]

    def _make_429(self, body: str) -> httpx.HTTPStatusError:
        response = httpx.Response(429, text=body)
        request = httpx.Request("POST", "http://test/api/chat")
        return httpx.HTTPStatusError(message="429", request=request, response=response)

    def test_quota_bodies_detected(self):
        for body in self.QUOTA_BODIES:
            exc = self._make_429(body)
            assert is_quota_exhausted(exc), f"Should detect quota in: {body[:50]}"

    def test_transient_bodies_not_detected(self):
        for body in self.TRANSIENT_BODIES:
            exc = self._make_429(body)
            assert not is_quota_exhausted(exc), f"Should NOT detect quota in: {body[:50]}"

    def test_non_429_not_quota(self):
        for status in (400, 401, 403, 500, 502, 503):
            response = httpx.Response(status, text="Monthly quota exceeded")
            request = httpx.Request("POST", "http://test/api/chat")
            exc = httpx.HTTPStatusError(message=str(status), request=request, response=response)
            assert not is_quota_exhausted(exc), f"Status {status} should not be quota"

    def test_non_http_errors_not_quota(self):
        exc = ValueError("not an HTTP error")
        assert not is_quota_exhausted(exc)

    def test_empty_response_body(self):
        exc = self._make_429("")
        assert not is_quota_exhausted(exc)

    def test_response_body_not_readable(self):
        response = httpx.Response(429)
        request = httpx.Request("POST", "http://test/api/chat")
        exc = httpx.HTTPStatusError(message="429", request=request, response=response)
        assert not is_quota_exhausted(exc)

    def test_all_quota_keywords_covered(self):
        """Every keyword in QUOTA_KEYWORDS should produce at least one match."""
        for kw in QUOTA_KEYWORDS:
            exc = self._make_429(f"This is a test for {kw} in the body")
            assert is_quota_exhausted(exc), f"Keyword not matched: {kw}"


# ---------------------------------------------------------------------------
# 2. Retryable exception checks
# ---------------------------------------------------------------------------

class TestIsRetryableWithQuota:
    """Verify is_retryable_exception returns False for quota 429s."""

    def test_transient_429_is_retryable(self):
        response = httpx.Response(429, text="Rate limit. Retry after 30s.")
        request = httpx.Request("POST", "http://test/api/chat")
        exc = httpx.HTTPStatusError(message="429", request=request, response=response)
        assert is_retryable_exception(exc)

    def test_quota_429_is_not_retryable(self):
        response = httpx.Response(429, text="Monthly quota exceeded.")
        request = httpx.Request("POST", "http://test/api/chat")
        exc = httpx.HTTPStatusError(message="429", request=request, response=response)
        assert not is_retryable_exception(exc)

    def test_500_always_retryable(self):
        response = httpx.Response(500, text="Server error")
        request = httpx.Request("POST", "http://test/api/chat")
        exc = httpx.HTTPStatusError(message="500", request=request, response=response)
        assert is_retryable_exception(exc)

    def test_400_not_retryable(self):
        response = httpx.Response(400, text="Bad request")
        request = httpx.Request("POST", "http://test/api/chat")
        exc = httpx.HTTPStatusError(message="400", request=request, response=response)
        assert not is_retryable_exception(exc)


# ---------------------------------------------------------------------------
# 3. QuotaExhaustedError is raised by retry_operation
# ---------------------------------------------------------------------------

class TestRetryOperationQuota:
    """Verify retry_operation raises QuotaExhaustedError for quota 429s."""

    @pytest.mark.asyncio
    async def test_quota_429_raises_quota_error_instantly(self):
        call_count = 0

        async def failing_op():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(429, text="Monthly quota exceeded.")
            request = httpx.Request("POST", "http://test/api/chat")
            raise httpx.HTTPStatusError(
                message="429", request=request, response=response
            )

        with pytest.raises(QuotaExhaustedError) as exc_info:
            await retry_operation(failing_op, max_retries=5, base_delay=0.0, max_delay=0.1)

        assert call_count == 1, (
            f"Should NOT retry quota 429, but got {call_count} calls"
        )
        assert "monthly quota" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_transient_429_is_retried(self):
        call_count = 0

        async def transient_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = httpx.Response(429, text="Rate limit. Retry after 30s.")
                request = httpx.Request("POST", "http://test/api/chat")
                raise httpx.HTTPStatusError(
                    message="429", request=request, response=response
                )
            return {"success": True}

        result = await retry_operation(transient_op, max_retries=5, base_delay=0.0, max_delay=0.1)

        assert result == {"success": True}
        assert call_count == 3, f"Should retry until success, got {call_count} calls"

    @pytest.mark.asyncio
    async def test_quota_error_carries_response(self):
        async def failing_op():
            response = httpx.Response(429, text="Daily limit reached.")
            request = httpx.Request("POST", "http://test/api/chat")
            raise httpx.HTTPStatusError(
                message="429", request=request, response=response
            )

        with pytest.raises(QuotaExhaustedError) as exc_info:
            await retry_operation(failing_op, max_retries=2, base_delay=0.0, max_delay=0.1)

        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 429


# ---------------------------------------------------------------------------
# 4. Circuit breaker records failure type
# ---------------------------------------------------------------------------

class TestCircuitBreakerFailureType:
    """Verify circuit breaker records and uses failure_type properly."""

    def test_default_failure_type_is_none(self):
        cb = CircuitBreaker("test")
        assert cb._last_failure_type is None

    @pytest.mark.asyncio
    async def test_record_failure_quota_sets_type(self):
        cb = CircuitBreaker("test")
        await cb.record_failure(failure_type="quota")
        assert cb._last_failure_type == "quota"

    @pytest.mark.asyncio
    async def test_record_failure_generic_sets_none(self):
        cb = CircuitBreaker("test")
        await cb.record_failure()
        assert cb._last_failure_type is None

    def test_stats_includes_failure_type_when_set(self):
        async def _run():
            cb = CircuitBreaker("test")
            await cb.record_failure(failure_type="quota")
            stats = cb.get_stats()
            assert stats["last_failure_type"] == "quota"
            return stats

        import asyncio
        stats = asyncio.run(_run())
        assert stats["last_failure_type"] == "quota"

    def test_stats_omits_failure_type_when_none(self):
        async def _run():
            cb = CircuitBreaker("test")
            await cb.record_failure()
            stats = cb.get_stats()
            assert "last_failure_type" not in stats or stats["last_failure_type"] is None
            return stats

        import asyncio
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Circuit breaker uses correct timeout for each failure type
# ---------------------------------------------------------------------------

class TestCircuitBreakerQuotaTimeout:
    """Verify the circuit breaker uses different timeouts for quota vs generic."""

    @pytest.mark.asyncio
    async def test_quota_failure_uses_longer_timeout(self):
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=1,
                reset_timeout=0.05,
                quota_reset_timeout=3600.0,
            ),
        )
        await cb.record_failure(failure_type="quota")
        assert cb.state == CircuitState.OPEN
        # Should NOT be allowed because quota timeout (3600s) hasn't elapsed
        assert not cb.is_call_allowed()

    @pytest.mark.asyncio
    async def test_generic_failure_uses_short_timeout(self):
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=1,
                reset_timeout=0.05,
                quota_reset_timeout=3600.0,
            ),
        )
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Wait for short timeout
        time.sleep(0.06)
        assert cb.is_call_allowed()
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_default_quota_timeout_is_one_hour(self):
        config = CircuitBreakerConfig(failure_threshold=5, reset_timeout=60.0)
        assert config.quota_reset_timeout == 3600.0

    @pytest.mark.asyncio
    async def test_quota_failure_opens_circuit_after_threshold(self):
        cb = CircuitBreaker(
            "quota-backend",
            CircuitBreakerConfig(
                failure_threshold=3,
                reset_timeout=60.0,
                quota_reset_timeout=7200.0,
            ),
        )
        for _ in range(3):
            await cb.record_failure(failure_type="quota")
        assert cb.state == CircuitState.OPEN
        stats = cb.get_stats()
        assert stats["last_failure_type"] == "quota"

    @pytest.mark.asyncio
    async def test_quota_and_generic_mixed_failures(self):
        cb = CircuitBreaker(
            "mixed-backend",
            CircuitBreakerConfig(
                failure_threshold=5,
                reset_timeout=0.05,
                quota_reset_timeout=3600.0,
            ),
        )
        # Mix of failure types - should use last failure's timeout
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_failure(failure_type="quota")
        await cb.record_failure()  # generic resets the last type
        assert cb.state == CircuitState.OPEN
        assert cb._last_failure_type is None  # last was generic
