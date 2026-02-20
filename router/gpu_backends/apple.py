"""Apple Silicon GPU backend implementation.

Apple Silicon (M1/M2/M3) uses unified memory architecture where
CPU and GPU share the same system RAM. There is no dedicated VRAM.

This backend estimates available GPU memory as a percentage of total
system RAM (default: 75%), which is the typical allocation for GPU
workloads in unified memory systems.

Detection:
    - Check if running on macOS with ARM64 architecture

Memory Query:
    - Estimate available as percentage of total RAM (configurable)
    - Estimate used memory via powermetrics (if available with sudo)
    - Fallback: Track estimated used memory via process accounting

Configuration:
    - If ROUTER_APPLE_UNIFIED_MEMORY_GB is set, use that as total RAM
    - Otherwise, auto-detect total system RAM
    - Default GPU allocation: 75% of total RAM
"""

import logging
import os
import platform
import re
import subprocess
import sys
from typing import List, Optional

from router.gpu_backends.base import GPUBackend, GPUMemory

logger = logging.getLogger(__name__)


class AppleBackend:
    """GPU backend for Apple Silicon Macs (M1/M2/M3).

    Estimates GPU memory based on unified memory architecture.
    Optionally uses powermetrics for detailed usage tracking.
    """

    def __init__(self, unified_memory_gb: Optional[float] = None):
        """Initialize Apple Silicon backend.

        Args:
            unified_memory_gb: Total unified memory in GB. If not provided,
                              auto-detects from system. Only needed if
                              auto-detection fails.
        """
        self._unified_memory_gb = unified_memory_gb
        self._chip_name: Optional[str] = None
        self._detect_chip()

        # Configuration: percentage of unified memory available for GPU
        self._gpu_memory_pct = 0.75

        # Track estimated GPU memory usage via process model loading
        self._estimated_used_gb = 0.0

    def _detect_chip(self):
        """Detect Apple chip type."""
        try:
            # Use sysctl to get machine info
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                self._chip_name = result.stdout.strip()
        except Exception as e:
            logger.debug(f"Failed to detect Apple chip: {e}")
            self._chip_name = "Apple Silicon"

    @property
    def vendor(self) -> str:
        return "apple"

    @property
    def device_name(self) -> str:
        if self._chip_name:
            return self._chip_name
        return "Apple Silicon"

    def is_available(self) -> bool:
        """Check if running on Apple Silicon."""
        is_macos = platform.system() == "Darwin"
        is_arm64 = platform.machine() == "arm64"
        return is_macos and is_arm64

    def get_total_ram_gb(self) -> float:
        """Get total system RAM in GB."""
        if self._unified_memory_gb:
            return self._unified_memory_gb

        try:
            # Use sysctl to get physical memory size
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                bytes_ram = int(result.stdout.strip())
                return bytes_ram / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to detect total RAM: {e}")

        # Common fallback values for Apple Silicon
        logger.warning(
            "Could not auto-detect total RAM. Common values: "
            "M1: 8/16GB, M2: 8/16/24GB, M3: 8/16/24/32GB, M2 Max: 32/64/96GB"
        )
        # Default fallback: 16GB (minimum for AI workloads)
        return 16.0

    def get_memory_info(self) -> List[GPUMemory]:
        """Get GPU memory info for Apple Silicon.

        Returns:
            List with single GPUMemory representing the integrated GPU.

        Note:
            Used memory is estimated. For accurate tracking, we rely on
            VRAMManager to track loaded models. Actual system usage may
            vary due to unified memory sharing with CPU.
        """
        total_ram = self.get_total_ram_gb()
        total_gb = total_ram * self._gpu_memory_pct

        # Try to get actual GPU usage from powermetrics if available
        used_gb = self._estimate_gpu_usage()

        free_gb = max(0.0, total_gb - used_gb)

        return [
            GPUMemory(
                index=0,
                total_gb=total_gb,
                used_gb=used_gb,
                free_gb=free_gb,
                vendor=self.vendor,
                device_name=f"{self.device_name} (Unified: {total_ram:.0f}GB)",
            )
        ]

    def _estimate_gpu_usage(self) -> float:
        """Estimate GPU memory usage.

        Strategy:
            1. If VRAMManager is tracking models, sum their estimated usage
            2. Optionally query powermetrics for actual GPU memory stats
            3. Fall back to tracked estimate

        Returns:
            Estimated GPU memory usage in GB
        """
        # TODO: Integrate with VRAMManager to get actual model memory
        # For now, return tracked estimate from model loading/unloading
        return self._estimated_used_gb

    def update_usage(self, used_gb: float):
        """Update estimated GPU usage (called by VRAMManager)."""
        self._estimated_used_gb = used_gb

    def _try_powermetrics(self) -> Optional[float]:
        """Try to get GPU usage from powermetrics (requires sudo)."""
        try:
            # powermetrics requires sudo on most systems
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "gpu_power", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse output for GPU memory usage
                # Example: "GPU Device memory usage: 4096 MB"
                for line in result.stdout.split("\n"):
                    if "GPU Device memory" in line.lower() or "GPU memory" in line.lower():
                        match = re.search(r"(\d+)\s*MB", line)
                        if match:
                            return int(match.group(1)) / 1024.0
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass
        except Exception as e:
            logger.debug(f"powermetrics query failed: {e}")

        return None
