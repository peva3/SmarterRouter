"""
Circuit breaker pattern for external service calls.

Prevents cascading failures by failing fast when a service is repeatedly failing.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"   # Normal operation, requests allowed
    OPEN = "open"       # Failing fast, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures before opening
    reset_timeout: float = 60.0         # Time in seconds before attempting half-open
    half_open_max_attempts: int = 3     # Number of successful attempts needed to close
    sliding_window_size: int = 100      # Max number of recent calls to track


class CircuitBreaker:
    """
    Circuit breaker implementation with sliding window.
    
    Tracks failures and successes within a sliding window of recent calls.
    When failure count exceeds threshold, circuit opens for reset_timeout.
    After timeout, circuit goes to half-open state, allowing limited test requests.
    If test requests succeed, circuit closes; otherwise reopens.
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_state_change_time = time.monotonic()
        self._lock = asyncio.Lock()
        
        # Sliding window of recent calls (True = success, False = failure)
        self._recent_calls: list[bool] = []
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    def is_call_allowed(self) -> bool:
        """
        Check if a call is allowed in current state.
        
        Returns False if circuit is OPEN.
        Returns True if CLOSED or HALF_OPEN (but HALF_OPEN has limited capacity).
        """
        if self._state == CircuitState.OPEN:
            # Check if reset timeout has elapsed
            if self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.reset_timeout:
                    # Transition to HALF_OPEN
                    self._set_state(CircuitState.HALF_OPEN)
                    return True
            return False
        return True
    
    async def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute a call within the circuit breaker.
        
        Args:
            func: Async callable to execute
            *args, **kwargs: Arguments to pass to func
        
        Returns:
            Result from func
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by func
        """
        if not self.is_call_allowed():
            raise CircuitBreakerOpenError(self.name, "Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise
    
    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._recent_calls.append(True)
            if len(self._recent_calls) > self.config.sliding_window_size:
                self._recent_calls.pop(0)
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.half_open_max_attempts:
                    self._set_state(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on consecutive successes
                if len(self._recent_calls) >= 3 and all(self._recent_calls[-3:]):
                    self._failure_count = 0
    
    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._recent_calls.append(False)
            if len(self._recent_calls) > self.config.sliding_window_size:
                self._recent_calls.pop(0)
            
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            
            if self._state == CircuitState.HALF_OPEN:
                # Immediate transition back to OPEN
                self._set_state(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._set_state(CircuitState.OPEN)
    
    def _set_state(self, new_state: CircuitState) -> None:
        """Update circuit state with callback."""
        old_state = self._state
        if old_state == new_state:
            return
        
        self._state = new_state
        self._last_state_change_time = time.monotonic()
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._failure_count = 0
            self._success_count = 0
        
        if self.on_state_change:
            try:
                self.on_state_change(self.name, old_state, new_state)
            except Exception:
                pass
    
    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "recent_calls_total": len(self._recent_calls),
            "recent_calls_failures": sum(1 for c in self._recent_calls if not c),
            "last_failure_time": self._last_failure_time,
            "last_state_change_time": self._last_state_change_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and call is rejected."""
    
    def __init__(self, breaker_name: str, message: str = "Circuit breaker is open"):
        super().__init__(f"{breaker_name}: {message}")
        self.breaker_name = breaker_name


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    async def reset(self, name: str | None = None) -> None:
        """Reset circuit breaker(s)."""
        async with self._lock:
            if name is None:
                for breaker in self._breakers.values():
                    breaker._set_state(CircuitState.CLOSED)
            elif name in self._breakers:
                self._breakers[name]._set_state(CircuitState.CLOSED)
    
    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}


# Global circuit breaker manager instance
_global_circuit_manager = CircuitBreakerManager()


async def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get a circuit breaker from the global manager."""
    return await _global_circuit_manager.get_breaker(name, config)


async def reset_circuit_breaker(name: str | None = None) -> None:
    """Reset circuit breaker(s) in the global manager."""
    await _global_circuit_manager.reset(name)