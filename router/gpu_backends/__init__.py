"""GPU backend manager that auto-detects and aggregates all available GPUs."""

import logging
from typing import List, Optional

from router.gpu_backends.base import GPUBackend, GPUMemory
from router.gpu_backends.nvidia import NVIDIABackend
from router.gpu_backends.amdgpu import AMDBackend
from router.gpu_backends.intel import IntelBackend
from router.gpu_backends.apple import AppleBackend

logger = logging.getLogger(__name__)


class GPUBackendManager:
    """Manages all available GPU backends and aggregates VRAM data.

    This manager:
    - Auto-detects all available GPU backends on initialization
    - Aggregates memory info from all vendors into a single list
    - Assigns global GPU indices (0, 1, 2, ...) across all vendors
    - Provides total VRAM calculation across all GPUs
    - Re-checks detection on every startup (catches driver/Docker fixes)

    Example:
        manager = GPUBackendManager(apple_unified_memory_gb=96)
        # Auto-detects NVIDIA + AMD + Intel + Apple if available

        all_gpus = manager.get_all_memory_info()
        # Returns [GPUMemory(index=0, ...), GPUMemory(index=1, ...), ...]

        total_vram = manager.get_total_vram()
        # Sum of all GPU memory in GB

        has_gpus = manager.has_gpus
        # True if any GPUs detected
    """

    def __init__(self, apple_unified_memory_gb: Optional[float] = None):
        """Initialize backend manager.

        Args:
            apple_unified_memory_gb: For Apple Silicon only - total unified memory in GB.
                                     If None, auto-detects from system.
        """
        self.backends: List[GPUBackend] = []
        self._apple_unified_memory_gb = apple_unified_memory_gb
        self._detect_all_backends()

    def _detect_all_backends(self) -> None:
        """Detect all available GPU backends.

        Tries each backend in order:
            NVIDIA, AMD, Intel, Apple

        Logs which backends are detected and which are skipped.
        Always logs a summary of detected GPUs (or warning if none).
        """
        backend_classes = [
            (NVIDIABackend, "NVIDIA"),
            (AMDBackend, "AMD"),
            (IntelBackend, "Intel"),
            (AppleBackend, "Apple Silicon"),
        ]

        detected_vendors = []

        for backend_class, vendor_name in backend_classes:
            try:
                # Pass config to Apple backend if needed
                if backend_class == AppleBackend:
                    backend = backend_class(unified_memory_gb=self._apple_unified_memory_gb)
                else:
                    backend = backend_class()

                if backend.is_available():
                    self.backends.append(backend)
                    detected_vendors.append(f"{backend.vendor}:{backend.device_name}")
                    logger.info(f"GPU backend detected: {backend.vendor} - {backend.device_name}")
                else:
                    logger.debug(f"GPU backend {vendor_name} not available")
            except Exception as e:
                logger.debug(f"GPU backend {vendor_name} check failed: {e}")

        if not self.backends:
            logger.warning(
                "No GPU backends detected. VRAM monitoring disabled. "
                "Possible causes: no GPU hardware, missing drivers, "
                "or incorrect Docker GPU configuration."
            )
        else:
            logger.info(
                f"Detected {len(self.backends)} GPU backend(s): {', '.join(detected_vendors)}"
            )

    def get_all_memory_info(self) -> List[GPUMemory]:
        """Get memory info from all GPUs across all vendors.

        Returns:
            List of GPUMemory objects with globally unique indices.
            GPU indices are assigned sequentially across all vendors:
                GPU 0 = first GPU from first available vendor
                GPU 1 = second GPU (from any vendor)
                etc.
        """
        all_gpus: List[GPUMemory] = []
        global_index = 0

        for backend in self.backends:
            try:
                gpus = backend.get_memory_info()
                for gpu in gpus:
                    # Assign global index
                    gpu.index = global_index
                    global_index += 1
                    all_gpus.append(gpu)
            except Exception as e:
                logger.warning(f"Error getting memory info from {backend.vendor} backend: {e}")
                # Continue with other backends
                continue

        return all_gpus

    def get_total_vram(self) -> float:
        """Get total VRAM across all detected GPUs.

        Returns:
            Total VRAM in gigabytes. Returns 0.0 if no GPUs detected.
        """
        gpus = self.get_all_memory_info()
        return sum(gpu.total_gb for gpu in gpus)

    def get_used_vram(self) -> float:
        """Get currently used VRAM across all detected GPUs.

        Returns:
            Used VRAM in gigabytes. Returns 0.0 if no GPUs detected.
        """
        gpus = self.get_all_memory_info()
        return sum(gpu.used_gb for gpu in gpus)

    def get_free_vram(self) -> float:
        """Get free/available VRAM across all detected GPUs.

        Returns:
            Free VRAM in gigabytes. Returns 0.0 if no GPUs detected.
        """
        gpus = self.get_all_memory_info()
        return sum(gpu.free_gb for gpu in gpus)

    @property
    def has_gpus(self) -> bool:
        """Check if any GPUs were detected."""
        return len(self.backends) > 0

    @property
    def gpu_count(self) -> int:
        """Get total number of detected GPUs across all vendors."""
        return len(self.get_all_memory_info())

    def get_vendor_info(self) -> List[str]:
        """Get list of detected vendor names for logging."""
        return [f"{backend.vendor}:{backend.device_name}" for backend in self.backends]
