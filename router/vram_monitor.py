"""VRAM monitoring and management for GPU memory."""

import subprocess
import re
import logging
from dataclasses import dataclass
from typing import Any, List, Optional
import asyncio
import time

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
    per_model_vram_gb: dict[str, float]  # Estimated per-model usage

    def to_log_string(self) -> str:
        """Format for concise application log."""
        if not self.per_model_vram_gb:
            return f"VRAM: {self.used_gb:.1f}/{self.total_gb:.1f}GB ({self.utilization_pct:.1f}%)"
        models_str = ", ".join(f"{m}:{v:.1f}GB" for m, v in self.per_model_vram_gb.items())
        return f"VRAM: {self.used_gb:.1f}/{self.total_gb:.1f}GB ({self.utilization_pct:.1f}%) models=[{models_str}]"


class VRAMMonitor:
    """
    Background task that periodically polls nvidia-smi and maintains a rolling buffer of metrics.

    Features:
    - Auto-detects NVIDIA GPU availability
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

        # GPU detection
        self.has_nvidia = self._check_nvidia_smi()
        self.gpu_name: Optional[str] = None

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi exists and returns valid data."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    self.gpu_name = parts[0].strip()
                return True
        except Exception:
            pass
        return False

    async def start(self):
        """Start background monitoring task."""
        if not self.has_nvidia:
            logger.info("VRAM monitor disabled: nvidia-smi not available")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"VRAM monitor started: interval={self.interval}s, GPU={self.gpu_name or 'NVIDIA'}"
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

            except Exception as e:
                logger.error(f"VRAM monitor error: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _sample(self) -> VRAMMetrics:
        """Take a single VRAM snapshot."""
        loop = asyncio.get_event_loop()
        stdout = await loop.run_in_executor(None, self._run_nvidia_smi)
        total_mb, used_mb, free_mb = self._parse_output(stdout)

        # Auto-detect total on first sample if not configured
        if self.total_vram_gb is None:
            self.total_vram_gb = total_mb / 1024
            logger.info(f"Auto-detected GPU VRAM: {self.total_vram_gb:.1f}GB")

        total_gb = self.total_vram_gb
        used_gb = used_mb / 1024
        free_gb = free_mb / 1024
        util_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0

        # Collect loaded models from app_state (via VRAMManager if available)
        models_loaded = []
        per_model_vram = {}
        if self.app_state:
            if hasattr(self.app_state, "vram_manager"):
                vm = self.app_state.vram_manager
                models_loaded = list(vm.loaded_models.keys())
                per_model_vram = dict(vm.loaded_models)
            elif hasattr(self.app_state, "loaded_models"):
                models_loaded = list(self.app_state.loaded_models.keys())
                per_model_vram = dict(self.app_state.loaded_models)

        return VRAMMetrics(
            timestamp=time.time(),
            total_gb=total_gb,
            used_gb=used_gb,
            free_gb=free_gb,
            utilization_pct=util_pct,
            models_loaded=models_loaded,
            per_model_vram_gb=per_model_vram,
        )

    def _run_nvidia_smi(self) -> str:
        """Execute nvidia-smi query (blocking)."""
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        result.check_returncode()
        return result.stdout

    def _parse_output(self, output: str) -> tuple[int, int, int]:
        """
        Parse nvidia-smi CSV output.
        Example: "24576 MiB, 12845 MiB, 11731 MiB"
        Returns: (total_mb, used_mb, free_mb)
        """
        line = output.strip().split("\n")[0]
        values = []
        for part in line.split(","):
            match = re.search(r"(\d+)", part.strip())
            if match:
                values.append(int(match.group(1)))
        if len(values) >= 3:
            return values[0], values[1], values[2]
        raise ValueError(f"Failed to parse nvidia-smi: {line}")

    # Public accessors
    def get_current(self) -> Optional[VRAMMetrics]:
        """Get the most recent metrics sample."""
        return self._samples[-1] if self._samples else None

    def get_history(self, minutes: int = 10) -> List[VRAMMetrics]:
        """Return samples from the last N minutes."""
        if not self._samples:
            return []
        cutoff = time.time() - (minutes * 60)
        return [s for s in self._samples if s.timestamp >= cutoff]
