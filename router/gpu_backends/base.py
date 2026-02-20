"""Base classes and protocols for GPU backends."""

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class GPUMemory:
    """Memory information for a single GPU.

    Attributes:
        index: Global index across all vendors (0, 1, 2, ...)
        total_gb: Total VRAM in gigabytes
        used_gb: Currently used VRAM in gigabytes
        free_gb: Available/free VRAM in gigabytes
        vendor: GPU vendor identifier ("nvidia", "amd", "intel", "apple")
        device_name: Human-readable GPU device name (e.g., "NVIDIA RTX 4090")
    """
    index: int
    total_gb: float
    used_gb: float
    free_gb: float
    vendor: str = ""
    device_name: str = ""


class GPUBackend(Protocol):
    """Protocol for GPU backend implementations.

    Each backend is responsible for detecting its own availability and
    providing VRAM memory information for all GPUs of that vendor on the system.

    Backends are queried by the GPUBackendManager which aggregates results
    from all available backends and assigns global GPU indices.
    """

    @property
    def vendor(self) -> str:
        """Return the GPU vendor identifier."""
        ...

    @property
    def device_name(self) -> str:
        """Return a human-readable device name for logging.

        For multi-GPU systems, this should return a generic name like
        "NVIDIA RTX 4090" rather than "GPU 0".
        """
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        This should perform lightweight detection (check for executable,
        driver files, sysfs entries) without actually querying VRAM data.

        Returns:
            True if the backend can provide VRAM monitoring on this system
        """
        ...

    def get_memory_info(self) -> List[GPUMemory]:
        """Get current VRAM memory information for all GPUs from this vendor.

        Returns:
            List of GPUMemory objects. The index field will be assigned
            by the GPUBackendManager, so initial value can be 0 or any placeholder.

        Raises:
            Exception: If unable to retrieve memory information (backend will
                       be marked as unavailable and skipped by manager)
        """
        ...
