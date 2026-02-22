"""VRAM monitoring and management for GPU memory."""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .gpu_backends.base import GPUMemory
from .gpu_backends import GPUBackendManager
from .metrics import VRAM_TOTAL_GB, VRAM_USED_GB, VRAM_UTILIZATION_PCT, gpu_metrics

logger = logging.getLogger(__name__)


@dataclass
class VRAMMetrics:
    """Snapshot of GPU VRAM usage at a point in time."""

    timestamp: float
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_pct: float
    models_loaded: List[str]  # From our tracking
    per_model_vram_gb: Dict[str, float]  # Estimated per-model usage
    gpus: List[GPUMemory]  # Per-GPU breakdown

    def to_log_string(self) -> str:
        """Format for concise application log."""
        if not self.per_model_vram_gb:
            base = f"VRAM: {self.used_gb:.1f}/{self.total_gb:.1f}GB ({self.utilization_pct:.1f}%)"
        else:
            models_str = ", ".join(f"{m}:{v:.1f}GB" for m, v in self.per_model_vram_gb.items())
            base = f"VRAM: {self.used_gb:.1f}/{self.total_gb:.1f}GB ({self.utilization_pct:.1f}%) models=[{models_str}]"
        # Add per-GPU info if multiple
        if len(self.gpus) > 1:
            gpu_details = ", ".join(
                f"{(g.vendor + ' ') if g.vendor else ''}{g.device_name if g.device_name else f'GPU{g.index}'}: {g.used_gb:.1f}/{g.total_gb:.1f}GB" for g in self.gpus
            )
            return f"{base} | {gpu_details}"
        elif self.gpus:
            # Single GPU: show vendor+name
            g = self.gpus[0]
            name = g.device_name if g.device_name else f"GPU{g.index}"
            return f"{base} | {g.vendor} {name}"
        return base


class VRAMMonitor:
    """
    Background task for monitoring GPU VRAM across all vendors.

    Features:
        - Auto-detects all available GPUs (NVIDIA, AMD, Intel, Apple)
        - Supports multi-GPU and mixed-vendor systems
        - Samples at configurable intervals
        - Keeps historical data for trend analysis
        - Logs concise summaries at separate interval
        - Provides current metrics and history retrieval
    """

    def __init__(
        self,
        interval: int = 30,
        total_vram_gb: Optional[float] = None,
        app_state: Optional[Any] = None,
        log_interval: int = 60,
        apple_unified_memory_gb: Optional[float] = None,
        amd_unified_memory_gb: Optional[float] = None,
    ):
        self.interval = interval
        self.total_vram_gb = total_vram_gb
        self.app_state = app_state
        self.log_interval = log_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._samples: List[VRAMMetrics] = []
        self._max_samples = 1000  # Keep ~30 min history at 30s interval
        self._last_log_time = 0

        # NEW: Multi-vendor GPU backend manager
        self.gpu_manager = GPUBackendManager(
            apple_unified_memory_gb=apple_unified_memory_gb,
            amd_unified_memory_gb=amd_unified_memory_gb,
        )
        self.has_gpu = self.gpu_manager.has_gpus

    @property
    def has_nvidia(self) -> bool:
        """Backward-compatible alias for has_gpu.
        
        Returns True if any GPU is detected (NVIDIA, AMD, Intel, or Apple Silicon).
        """
        return self.has_gpu

    @has_nvidia.setter
    def has_nvidia(self, value: bool):
        """Backward-compatible setter for has_nvidia.
        
        This allows tests to set the value directly.
        """
        self.has_gpu = value

    @property
    def gpu_name(self) -> str:
        """Backward-compatible property for GPU name.
        
        Returns the name of the first detected GPU or empty string.
        """
        # Check cached name from nvidia-smi check
        if hasattr(self, '_cached_gpu_name') and self._cached_gpu_name:
            return self._cached_gpu_name
        if self.gpu_manager.backends:
            for backend in self.gpu_manager.backends:
                if backend.is_available():
                    return backend.device_name
        return ""

    @gpu_name.setter
    def gpu_name(self, value: str):
        """Backward-compatible setter for gpu_name (for tests)."""
        self._cached_gpu_name = value

    async def start(self):
        """Start background monitoring task."""
        if not self.has_gpu:
            logger.warning(
                "No GPU backends detected. VRAM monitoring disabled. "
                "Possible causes: no GPU hardware, missing drivers, "
                "or incorrect Docker GPU configuration."
            )
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

        total_vram = self.gpu_manager.get_total_vram()
        vendor_info = self.gpu_manager.get_vendor_info()
        logger.info(
            f"VRAM monitor started: interval={self.interval}s, "
            f"total={total_vram:.1f}GB across {len(self.gpu_manager.backends)} vendor(s)"
        )

    async def stop(self):
        """Stop monitoring gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Main sampling loop."""
        while self._running:
            try:
                metrics = await self._sample()
                self._samples.append(metrics)
                if len(self._samples) > self._max_samples:
                    self._samples.pop(0)

                # Periodic logging
                now = time.time()
                if now - self._last_log_time >= self.log_interval:
                    logger.info(f"[VRAM] {metrics.to_log_string()}")
                    self._last_log_time = now

                    # Check thresholds
                    if metrics.utilization_pct >= 95:
                        logger.error(
                            f"VRAM CRITICAL: {metrics.utilization_pct:.1f}% used - risk of OOM"
                        )
                    elif metrics.utilization_pct >= 85:
                        logger.warning(f"VRAM high: {metrics.utilization_pct:.1f}% used")

                # Update Prometheus metrics
                VRAM_TOTAL_GB.set(metrics.total_gb)
                VRAM_USED_GB.set(metrics.used_gb)
                VRAM_UTILIZATION_PCT.set(metrics.utilization_pct)
                # Update per-GPU metrics with vendor label
                for gpu in metrics.gpus:
                    idx_str = str(gpu.index)
                    gpu_metrics["total"].labels(gpu_index=idx_str, vendor=gpu.vendor).set(
                        gpu.total_gb
                    )
                    gpu_metrics["used"].labels(gpu_index=idx_str, vendor=gpu.vendor).set(
                        gpu.used_gb
                    )
                    gpu_metrics["free"].labels(gpu_index=idx_str, vendor=gpu.vendor).set(
                        gpu.free_gb
                    )

            except Exception as e:
                logger.error(f"VRAM monitor error: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _sample(self) -> VRAMMetrics:
        """Take a single VRAM snapshot."""
        loop = asyncio.get_event_loop()
        gpus = []
        
        # Backward compatibility: check if _run_nvidia_smi is set (for tests)
        if self._run_nvidia_smi is not None:
            try:
                output = self._run_nvidia_smi()
                if output:
                    _, _, _, gpus = self._parse_output(output)
            except Exception as e:
                logger.debug(f"Legacy nvidia-smi query failed: {e}")
        
        # Normal path: use GPU backend manager
        if not gpus:
            try:
                # Get memory info from all GPUs via manager
                gpus = await loop.run_in_executor(None, self.gpu_manager.get_all_memory_info)
            except Exception as e:
                logger.error(f"Failed to get GPU memory info: {e}")
                gpus = []

        if not gpus:
            # No GPU data available - return zeroed metrics
            if self.total_vram_gb:
                total_gb = self.total_vram_gb
                used_gb = 0.0
                free_gb = total_gb
                util_pct = 0.0
            else:
                total_gb = used_gb = free_gb = util_pct = 0.0

            return VRAMMetrics(
                timestamp=time.time(),
                total_gb=total_gb,
                used_gb=used_gb,
                free_gb=free_gb,
                utilization_pct=util_pct,
                models_loaded=[],
                per_model_vram_gb={},
                gpus=[],
            )

        # Auto-detect total on first sample if not configured
        if self.total_vram_gb is None:
            self.total_vram_gb = sum(g.total_gb for g in gpus)
            logger.info(f"Auto-detected total VRAM: {self.total_vram_gb:.1f}GB")

        total_gb = sum(g.total_gb for g in gpus)
        used_gb = sum(g.used_gb for g in gpus)
        free_gb = sum(g.free_gb for g in gpus)
        util_pct = (used_gb / total_gb) * 100 if total_gb > 0 else 0

        # Collect loaded models from app_state (via VRAMManager if available)
        models_loaded = []
        per_model_vram = {}
        if self.app_state and hasattr(self.app_state, "vram_manager"):
            vm = self.app_state.vram_manager
            models_loaded = list(vm.loaded_models.keys())
            per_model_vram = dict(vm.loaded_models)

        return VRAMMetrics(
            timestamp=time.time(),
            total_gb=total_gb,
            used_gb=used_gb,
            free_gb=free_gb,
            utilization_pct=util_pct,
            models_loaded=models_loaded,
            per_model_vram_gb=per_model_vram,
            gpus=gpus,
        )

    # Public accessors remain unchanged
    def get_current(self) -> Optional[VRAMMetrics]:
        """Get the most recent metrics sample."""
        return self._samples[-1] if self._samples else None

    def get_history(self, minutes: int = 10) -> List[VRAMMetrics]:
        """Return samples from the last N minutes."""
        if not self._samples:
            return []
        cutoff = time.time() - (minutes * 60)
        return [s for s in self._samples if s.timestamp >= cutoff]

    # Backward compatibility methods for tests
    _run_nvidia_smi: Optional[Callable[[], str]] = None

    def _parse_output(self, output: str) -> tuple:
        """Backward-compatible method to parse nvidia-smi output.
        
        This method is deprecated and provided for test compatibility.
        The new implementation uses GPUBackendManager for memory queries.
        """
        import re
        gpus = []
        
        # Parse old nvidia-smi format: "0, 24576 MiB, 12845 MiB, 11731 MiB"
        # or multi-GPU: "0, 24576 MiB, 12845 MiB, 11731 MiB\n1, 24576 MiB, 8000 MiB, 16576 MiB"
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            match = re.match(r"(\d+),\s*(\d+)\s*MiB,\s*(\d+)\s*MiB,\s*(\d+)\s*MiB", line)
            if match:
                idx, total, used, free = match.groups()
                gpus.append(GPUMemory(
                    index=int(idx),
                    total_gb=int(total) / 1024,
                    used_gb=int(used) / 1024,
                    free_gb=int(free) / 1024,
                    vendor="nvidia",
                    device_name=f"GPU{idx}",
                ))
        
        if not gpus:
            raise ValueError(f"Could not parse nvidia-smi output: {output}")
        
        total_mb = sum(int(g.total_gb * 1024) for g in gpus)
        used_mb = sum(int(g.used_gb * 1024) for g in gpus)
        free_mb = sum(int(g.free_gb * 1024) for g in gpus)
        
        return total_mb, used_mb, free_mb, gpus

    def _check_nvidia_smi_available(self) -> bool:
        """Backward-compatible method to check nvidia-smi availability.
        
        This method is deprecated. The new implementation uses GPUBackendManager
        which auto-detects all GPU vendors.
        """
        # Check has_nvidia first for test compatibility (tests set this property)
        if self.has_nvidia:
            return True
        # Try actual backends
        if self.gpu_manager.has_gpus and any(b.vendor == "nvidia" for b in self.gpu_manager.backends):
            return True
        # Fallback: try running nvidia-smi directly (for backward compatibility)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Extract GPU name
                first_line = result.stdout.strip().split("\n")[0]
                self._cached_gpu_name = first_line.split(",")[0].strip()
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            pass
        return False

    def _check_nvidia_smi(self) -> bool:
        """Alias for _check_nvidia_smi_available for test compatibility."""
        return self._check_nvidia_smi_available()
