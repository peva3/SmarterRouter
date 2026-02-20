"""Intel GPU backend implementation for dedicated GPUs (Arc series).

This backend only monitors dedicated Intel Arc GPUs with their own VRAM.
Integrated graphics (Intel UHD, Iris, etc.) are skipped as they use
shared system memory and are not suitable for AI workloads.

Detection relies on:
    - sysfs entries for dedicated memory (lmem_total)
    - Optional: intel_gpu_top command for real-time stats

Supported Hardware:
    - Intel Arc A-series (A380, A770, etc.)
    - Intel Data Center GPU (Flex, Max series)
"""

import json
import logging
import os
import re
import subprocess
from typing import List, Optional

from router.gpu_backends.base import GPUBackend, GPUMemory

logger = logging.getLogger(__name__)

# Intel vendor ID
INTEL_VENDOR_ID = "0x8086"


class IntelBackend:
    """GPU backend for Intel dedicated GPUs (Arc series only)."""

    def __init__(self, intel_gpu_top_path: str = "intel_gpu_top"):
        """Initialize Intel backend.

        Args:
            intel_gpu_top_path: Path to intel_gpu_top executable (default: "intel_gpu_top")
        """
        self._intel_gpu_top_path = intel_gpu_top_path
        self._sysfs_cards: List[str] = []
        self._detect_sysfs_cards()

    def _detect_sysfs_cards(self):
        """Detect Intel DRM cards in sysfs."""
        try:
            for card in os.listdir("/sys/class/drm"):
                if card.startswith("card"):
                    device_path = f"/sys/class/drm/{card}/device"
                    vendor_path = f"{device_path}/vendor"
                    if os.path.exists(vendor_path):
                        try:
                            with open(vendor_path, "r") as f:
                                vendor = f.read().strip()
                                if vendor == INTEL_VENDOR_ID:
                                    self._sysfs_cards.append(device_path)
                        except (IOError, OSError):
                            continue
        except (FileNotFoundError, OSError):
            pass

    def _is_dedicated_gpu(self, device_path: str) -> bool:
        """Check if this Intel GPU has dedicated VRAM (lmem).

        Integrated graphics will NOT have lmem_total or will have 0.
        Dedicated Arc GPUs will have non-zero lmem_total.

        Args:
            device_path: Path to /sys/class/drm/cardX/device

        Returns:
            True if this is a dedicated GPU with its own VRAM
        """
        lmem_total_path = f"{device_path}/lmem_total"
        if not os.path.exists(lmem_total_path):
            return False

        try:
            with open(lmem_total_path, "r") as f:
                content = f.read().strip()
                match = re.search(r"(\d+)", content)
                if match:
                    total_bytes = int(match.group(1))
                    return total_bytes > 0
        except (IOError, OSError, ValueError):
            pass

        return False

    @property
    def vendor(self) -> str:
        return "intel"

    @property
    def device_name(self) -> str:
        return "Intel Arc GPU"

    def is_available(self) -> bool:
        """Check if any dedicated Intel GPUs are available.

        Detection order:
            1. Try intel_gpu_top (if installed)
            2. Check sysfs for dedicated GPUs with lmem_total > 0

        Returns:
            True if at least one dedicated Intel GPU is detected
        """
        # Try intel_gpu_top first
        try:
            result = subprocess.run(
                [self._intel_gpu_top_path, "-J"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                # Parse JSON output to check for any engines
                import json
                data = json.loads(result.stdout)
                if "cards" in data and data["cards"]:
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass
        except Exception as e:
            logger.debug(f"intel_gpu_top check failed: {e}")

        # Check sysfs for dedicated GPUs
        for device_path in self._sysfs_cards:
            if self._is_dedicated_gpu(device_path):
                return True

        return False

    def get_memory_info(self) -> List[GPUMemory]:
        """Get VRAM memory info for all dedicated Intel GPUs.

        Returns:
            List of GPUMemory objects, one per detected Intel Arc GPU.

        Raises:
            ValueError: If no GPU data could be obtained
        """
        gpus: List[GPUMemory] = []

        # Try intel_gpu_top method (preferred)
        try:
            return self._query_intel_gpu_top()
        except Exception as e:
            logger.debug(f"intel_gpu_top query failed, trying sysfs: {e}")

        # Fallback to sysfs
        for device_path in self._sysfs_cards:
            if self._is_dedicated_gpu(device_path):
                try:
                    gpu = self._query_sysfs_device(device_path)
                    if gpu:
                        gpus.append(gpu)
                except Exception as e:
                    logger.debug(f"Failed to query Intel GPU at {device_path}: {e}")
                    continue

        if not gpus:
            raise ValueError("No Intel Arc GPU data available")

        return gpus

    def _query_intel_gpu_top(self) -> List[GPUMemory]:
        """Query Intel GPUs using intel_gpu_top JSON output."""
        try:
            result = subprocess.run(
                [self._intel_gpu_top_path, "-J"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result.check_returncode()

            import json
            data = json.loads(result.stdout)
            gpus: List[GPUMemory] = []

            cards = data.get("cards", [])
            for card in cards:
                # intel_gpu_top provides engine info but not direct VRAM
                # We need to fall back to sysfs for VRAM totals
                # Here we just extract card ID and will get VRAM from sysfs
                card_id = card.get("card", 0)
                device_path = f"/sys/class/drm/card{card_id}/device"
                if os.path.exists(device_path) and self._is_dedicated_gpu(device_path):
                    gpu = self._query_sysfs_device(device_path)
                    if gpu:
                        gpus.append(gpu)

            if not gpus:
                raise ValueError("No Intel GPU data from intel_gpu_top")

            return gpus

        except json.JSONDecodeError as e:
            logger.warning(f"intel_gpu_top JSON parsing failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"intel_gpu_top query error: {e}")
            raise

    def _query_sysfs_device(self, device_path: str) -> Optional[GPUMemory]:
        """Query VRAM info from sysfs for a single Intel GPU device.

        Args:
            device_path: Path to /sys/class/drm/cardX/device

        Returns:
            GPUMemory object, or None if data incomplete
        """
        try:
            # Required: lmem_total (dedicated memory in bytes)
            total_path = f"{device_path}/lmem_total"
            if not os.path.exists(total_path):
                return None

            with open(total_path, "r") as f:
                total_str = f.read().strip()
                match = re.search(r"(\d+)", total_str)
                if not match:
                    return None
                total_bytes = int(match.group(1))
                total_gb = total_bytes / (1024 ** 3)

            # Optional: lmem_used (used memory in bytes)
            used_gb = 0.0
            used_path = f"{device_path}/lmem_used"
            if os.path.exists(used_path):
                with open(used_path, "r") as f:
                    used_str = f.read().strip()
                    match = re.search(r"(\d+)", used_str)
                    if match:
                        used_bytes = int(match.group(1))
                        used_gb = used_bytes / (1024 ** 3)

            # Compute free memory (sysfs doesn't always have lmem_free)
            free_gb = max(0.0, total_gb - used_gb)

            # Get card number for device name
            card_match = re.search(r"card(\d+)", device_path)
            card_id = card_match.group(1) if card_match else "?"

            return GPUMemory(
                index=0,
                total_gb=total_gb,
                used_gb=used_gb,
                free_gb=free_gb,
                vendor=self.vendor,
                device_name=f"Intel Arc GPU (card {card_id})",
            )
        except (IOError, OSError, ValueError) as e:
            logger.debug(f"Failed to read Intel GPU sysfs at {device_path}: {e}")
            return None
