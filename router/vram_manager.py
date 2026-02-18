"""VRAM allocation and model lifecycle management."""

import logging
from typing import Any, Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class VRAMExceededError(Exception):
    """Raised when VRAM budget is exceeded and cannot free enough memory."""

    pass


class VRAMManager:
    """
    Coordinates model loading/unloading to stay within VRAM budget.

    Responsibilities:
    - Track loaded models and their estimated VRAM usage
    - Check if new model fits in available budget
    - Proactively unload models (LRU/largest) to make room
    - Enforce pinned model constraint (never unload)
    - Integrate with backend for actual load/unload operations
    """

    # Internal safety buffer for fragmentation/measurement errors (not user-configurable)
    FRAGMENTATION_BUFFER_GB = 0.5

    def __init__(
        self,
        max_vram_gb: float,
        auto_unload_enabled: bool = True,
        unload_strategy: str = "lru",
        monitor: Optional[Any] = None,
    ):
        """
        Initialize VRAM manager.

        Args:
            max_vram_gb: Maximum VRAM to allocate for models (user configured)
            auto_unload_enabled: If True, automatically unload models when space needed
            unload_strategy: "lru" (least recently used) or "largest" (biggest first)
            monitor: Optional VRAMMonitor instance for real-time usage checks
        """
        self.max_vram = max_vram_gb
        self.auto_unload = auto_unload_enabled
        self.unload_strategy = unload_strategy
        self.monitor = monitor

        # State
        self.loaded_models: Dict[str, float] = {}  # model_name -> estimated_vram_gb
        self.pinned_model: Optional[str] = None
        self._backend_ref: Optional[Any] = None  # LLMBackend instance, set by RouterEngine
        
        # Concurrency control
        self._lock = asyncio.Lock()

    def set_backend(self, backend: Any):
        """Called by RouterEngine to provide backend for load/unload operations."""
        self._backend_ref = backend

    def get_available_vram(self) -> float:
        """
        Calculate available VRAM for loading new models.

        Formula: available = effective_budget - sum(loaded_estimates)
        where effective_budget = max_vram - FRAGMENTATION_BUFFER_GB

        This does NOT query nvidia-smi; it's based on our allocation tracking.
        
        Note: This is a read-only operation; lock acquired by callers when needed.
        """
        effective_budget = self.max_vram - self.FRAGMENTATION_BUFFER_GB
        allocated = sum(self.loaded_models.values())
        available = effective_budget - allocated
        return max(available, 0.0)

    def can_load(self, model_name: str, vram_estimate_gb: float) -> bool:
        """Check if model can be loaded within current VRAM budget."""
        return self.get_available_vram() >= vram_estimate_gb

    def get_current_allocated(self) -> float:
        """Return total VRAM currently allocated to loaded models."""
        return sum(self.loaded_models.values())

    def get_utilization_pct(self) -> float:
        """Return percentage of max_vram currently allocated."""
        if self.max_vram <= 0:
            return 0.0
        return (self.get_current_allocated() / self.max_vram) * 100

    async def load_model(
        self,
        model_name: str,
        vram_estimate_gb: float,
        pin: bool = False,
    ):
        """
        Load a model, managing VRAM constraints.

        Args:
            model_name: Name of model to load
            vram_estimate_gb: Estimated VRAM requirement (from DB profile)
            pin: If True, mark as pinned (never auto-unloaded)

        Raises:
            VRAMExceededError: If model cannot be loaded even after unloads
        """
        async with self._lock:
            if model_name in self.loaded_models:
                logger.debug(f"Model {model_name} already loaded")
                return

            if not self._backend_ref:
                raise RuntimeError("VRAMManager: backend not set. Call set_backend() first.")

            # Check if fits
            if not self.can_load(model_name, vram_estimate_gb):
                needed = vram_estimate_gb - self.get_available_vram()
                logger.info(f"VRAM: Need {needed:.1f}GB more for {model_name}, triggering unload...")
                if self.auto_unload:
                    await self._free_vram(needed)
                else:
                    raise VRAMExceededError(
                        f"Insufficient VRAM for {model_name} (need {vram_estimate_gb:.1f}GB, "
                        f"available {self.get_available_vram():.1f}GB) and auto_unload disabled"
                    )

                # Re-check after unload attempts
                if not self.can_load(model_name, vram_estimate_gb):
                    raise VRAMExceededError(
                        f"Still cannot load {model_name} after unload attempts. "
                        f"Need {vram_estimate_gb:.1f}GB, available {self.get_available_vram():.1f}GB"
                    )

            # Load the model via backend
            logger.info(f"Loading model {model_name} (~{vram_estimate_gb:.1f}GB)")
            try:
                await self._backend_ref.load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

            self.loaded_models[model_name] = vram_estimate_gb

            if pin:
                self.pinned_model = model_name
                logger.info(f"Pinned model {model_name} (will not auto-unload)")

    async def _unload_model_internal(self, model_name: str):
        """Internal version of unload_model - caller must hold self._lock."""
        if model_name not in self.loaded_models:
            return

        if model_name == self.pinned_model:
            logger.warning(f"VRAM: Attempted to unload pinned model {model_name} - skipping")
            return

        if not self._backend_ref:
            raise RuntimeError("VRAMManager: backend not set")

        logger.info(f"Unloading model {model_name}")
        try:
            await self._backend_ref.unload_model(model_name)
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            # Still remove from tracking even if unload failed
        
        # Remove from tracking
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

    async def unload_model(self, model_name: str):
        """Unload a specific model (public API - acquires lock)."""
        async with self._lock:
            await self._unload_model_internal(model_name)

    async def _free_vram(self, needed_gb: float):
        """
        Free VRAM by unloading models according to strategy.
        
        Note: This method assumes the caller holds self._lock to prevent race conditions.
        It is called from load_model() which already acquires the lock.

        Args:
            needed_gb: Amount of VRAM to free (in GB)

        Raises:
            VRAMExceededError: If cannot free enough VRAM
        """
        if not self.loaded_models:
            raise VRAMExceededError(f"Need {needed_gb:.1f}GB but no models are loaded")

        # Build candidate list (exclude pinned)
        candidates = [(m, vram) for m, vram in self.loaded_models.items() if m != self.pinned_model]

        if not candidates:
            raise VRAMExceededError(f"Need {needed_gb:.1f}GB but only pinned models are loaded")

        # Sort according to strategy
        if self.unload_strategy == "largest":
            candidates.sort(key=lambda x: x[1], reverse=True)  # Largest first (most VRAM)
        else:  # "lru" (default)
            # To implement true LRU, we'd need access to recent_selections from cache.
            # For now, fall back to largest-first as a reasonable heuristic.
            candidates.sort(key=lambda x: x[1], reverse=True)

        freed = 0.0
        unloaded_models = []
        for model_name, vram in candidates:
            if freed >= needed_gb:
                break
            try:
                # Call the internal version to avoid re-acquiring lock
                await self._unload_model_internal(model_name)
                freed += vram
                unloaded_models.append(model_name)
            except Exception as e:
                logger.error(f"Failed to unload {model_name}: {e}")

        if freed < needed_gb:
            raise VRAMExceededError(
                f"Could only free {freed:.1f}GB of {needed_gb:.1f}GB needed. "
                f"Unloaded: {unloaded_models}"
            )

        logger.info(
            f"VRAM: Freed {freed:.1f}GB by unloading {len(unloaded_models)} models: {unloaded_models}"
        )

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        return model_name in self.loaded_models

    def get_loaded_models(self) -> List[str]:
        """Return list of currently loaded model names."""
        return list(self.loaded_models.keys())

    def get_vram_usage_by_model(self) -> Dict[str, float]:
        """Return dict of model -> estimated VRAM usage (GB)."""
        return dict(self.loaded_models)

    def should_trigger_unload(self) -> bool:
        """
        Check if current VRAM utilization exceeds unload threshold.
        Used for proactive unload before loading new models.
        """
        if not self.auto_unload:
            return False
        util_pct = self.get_utilization_pct()
        # Check against configured threshold (need to get from settings)
        # This will be called from RouterEngine which has access to settings
        return False  # Placeholder - actual check uses settings
