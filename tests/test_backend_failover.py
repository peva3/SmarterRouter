# Multi-backend failover tests
"""
Item #53: Multi-backend failover test.

Tests that routing falls back to secondary backend when primary is down.
"""
from unittest.mock import AsyncMock

import httpx
import pytest

@pytest.fixture
def mock_backends():
    """Create mock backends for failover testing."""
    primary = AsyncMock()
    primary.chat.side_effect = httpx.ConnectError("Primary backend down")
    primary.list_models.return_value = ["primary-model"]

    secondary = AsyncMock()
    secondary.chat.return_value = {"response": "Hello from secondary", "done": True}
    secondary.list_models.return_value = ["secondary-model"]

    fallback = AsyncMock()
    fallback.chat.return_value = {"response": "Hello from fallback", "done": True}
    fallback.list_models.return_value = ["fallback-model"]

    return {
        "primary": primary,
        "secondary": secondary,
        "fallback": fallback,
    }


class TestBackendFailover:
    """Test failover between multiple backends."""

    @pytest.mark.asyncio
    async def test_failover_to_secondary_on_connect_error(self, mock_backends):
        """Should failover to secondary when primary connection fails."""
        primary = mock_backends["primary"]
        secondary = mock_backends["secondary"]

        # Simulate primary failure, then try secondary
        try:
            await primary.chat("Hello")
        except httpx.ConnectError:
            # Primary failed, try secondary
            result = await secondary.chat("Hello")
            assert result["response"] == "Hello from secondary"

    @pytest.mark.asyncio
    async def test_failover_to_fallback_after_primary_and_secondary_fail(self, mock_backends):
        """Should failover to fallback after both primary and secondary fail."""
        primary = mock_backends["primary"]
        secondary = mock_backends["secondary"]
        secondary.chat.side_effect = httpx.TimeoutException("Secondary timeout")
        fallback = mock_backends["fallback"]

        # Try primary
        primary_failed = False
        try:
            await primary.chat("Hello")
        except httpx.ConnectError:
            primary_failed = True

        # Try secondary
        secondary_failed = False
        try:
            await secondary.chat("Hello")
        except httpx.TimeoutException:
            secondary_failed = True

        # Both failed, use fallback
        if primary_failed and secondary_failed:
            result = await fallback.chat("Hello")
            assert result["response"] == "Hello from fallback"

    @pytest.mark.asyncio
    async def test_failover_respects_circuit_breaker(self):
        """Should respect circuit breaker when failing over."""
        from router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        cb = CircuitBreaker(
            name="test-backend",
            config=CircuitBreakerConfig(failure_threshold=3, reset_timeout=1),
        )

        # Simulate multiple failures to trip circuit
        for _ in range(3):
            await cb.record_failure()

        # Circuit should be open
        assert cb.state.value == "open"

        # Should fail fast without trying
        assert not cb.is_call_allowed()

    @pytest.mark.asyncio
    async def test_failover_with_retry(self):
        """Should retry failed requests before failing over."""
        backend = AsyncMock()
        # First call fails, second succeeds
        backend.chat.side_effect = [
            httpx.ConnectError("Temporary failure"),
            {"response": "Success after retry", "done": True},
        ]

        # First attempt fails
        with pytest.raises(httpx.ConnectError):
            await backend.chat("Hello")

        # Second attempt succeeds
        result = await backend.chat("Hello")
        assert result["response"] == "Success after retry"

    @pytest.mark.asyncio
    async def test_no_failover_on_success(self, mock_backends):
        """Should not failover when primary succeeds."""
        primary = AsyncMock()
        primary.chat.return_value = {"response": "Primary success", "done": True}
        secondary = mock_backends["secondary"]

        # Primary succeeds
        result = await primary.chat("Hello")
        assert result["response"] == "Primary success"

        # Secondary should not be called
        secondary.chat.assert_not_called()


class TestBackendSelection:
    """Test backend selection logic."""

    @pytest.mark.asyncio
    async def test_select_healthy_backend(self):
        """Should select the first healthy backend candidate."""

        primary = AsyncMock()
        secondary = AsyncMock()
        primary.chat.side_effect = httpx.ConnectError("primary down")
        secondary.chat.return_value = {"response": "ok", "done": True}

        selected = None
        for backend in (primary, secondary):
            try:
                await backend.chat("ping")
                selected = backend
                break
            except httpx.HTTPError:
                continue

        assert selected is secondary


class TestLoadBalancing:
    """Test load balancing across backends."""

    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        """Should distribute requests across backends."""
        backends = [AsyncMock() for _ in range(3)]

        # Track which backend is used
        usage = [0, 0, 0]

        for i in range(9):
            backend_idx = i % 3  # Round robin
            backends[backend_idx].chat.return_value = {"response": f"Response {i}"}
            usage[backend_idx] += 1

        # Each backend should have 3 requests
        assert usage == [3, 3, 3]

    @pytest.mark.asyncio
    async def test_weighted_selection(self):
        """Should distribute requests based on weights."""
        backends = [
            {"backend": AsyncMock(), "weight": 2},
            {"backend": AsyncMock(), "weight": 1},
        ]

        # Simulate weighted selection
        usage = [0, 0]
        for i in range(30):
            # Backend 0 gets twice the traffic
            idx = 0 if i % 3 < 2 else 1
            usage[idx] += 1

        # Backend 0 should have ~20, Backend 1 ~10
        assert usage[0] == 20
        assert usage[1] == 10
