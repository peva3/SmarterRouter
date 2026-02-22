"""Intel GPU backend implementation for dedicated GPUs (Arc series).

This backend monitors dedicated Intel GPUs with their own VRAM:
    - Intel Arc A-series (A380, A770, etc.) - uses i915 driver
    - Intel Arc B-series (B580, etc.) - uses xe driver  
    - Intel Data Center GPU (Flex, Max series)

Integrated graphics (Intel UHD, Iris Xe, Arc 130V/140V iGPU) are skipped
as they use shared system memory and are not suitable for LLM workloads.

Detection Methods:
    - i915 driver: sysfs lmem_total/lmem_used
    - xe driver: fdinfo drm-total-vram0 (newer Battlemage GPUs)

Note: Intel Core Ultra (Meteor Lake/Lunar Lake) with Arc 130V/140V iGPUs
use shared system memory and are NOT detected by this backend.
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
        """Check if this Intel GPU has dedicated VRAM.

        Detection methods:
            - i915 driver: lmem_total > 0
            - xe driver: Check driver symlink

        Integrated graphics (UHD, Iris Xe, Arc 130V/140V iGPU) will NOT
        have lmem_total or will have 0, and they use i915 driver.

        Args:
            device_path: Path to /sys/class/drm/cardX/device

        Returns:
            True if this is a dedicated GPU with its own VRAM
        """
        driver_path = f"{device_path}/driver"
        try:
            driver_link = os.readlink(driver_path)
            if "xe" in driver_link:
                return True
        except (OSError, IOError):
            pass

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

        Supports both i915 driver (lmem_total) and xe driver (fdinfo).

        Args:
            device_path: Path to /sys/class/drm/cardX/device

        Returns:
            GPUMemory object, or None if data incomplete
        """
        card_match = re.search(r"card(\d+)", device_path)
        card_id = card_match.group(1) if card_match else "?"

        xe_result = self._query_xe_driver(device_path, card_id)
        if xe_result:
            return xe_result

        return self._query_i915_device(device_path, card_id)

    def _query_xe_driver(self, device_path: str, card_id: str) -> Optional[GPUMemory]:
        """Query VRAM for xe driver (Battlemage, Xe2).

        The xe driver uses fdinfo format instead of lmem_total.
        We read from /sys/class/drm/cardX/device/drm/ for vram info.

        Args:
            device_path: Path to /sys/class/drm/cardX/device
            card_id: Card number string

        Returns:
            GPUMemory object, or None if not xe driver
        """
        driver_path = f"{device_path}/driver"
        try:
            driver_link = os.readlink(driver_path)
            if "xe" not in driver_link:
                return None
        except (OSError, IOError):
            return None

        logger.debug(f"Intel GPU card {card_id} uses xe driver")

        vram_paths = [
            f"/sys/class/drm/card{card_id}/device/tile0/vram_total",
            f"{device_path}/tile0/vram_total",
        ]

        total_bytes = None
        for vram_path in vram_paths:
            if os.path.exists(vram_path):
                try:
                    with open(vram_path, "r") as f:
                        content = f.read().strip()
                        match = re.search(r"(\d+)", content)
                        if match:
                            total_bytes = int(match.group(1))
                            break
                except (IOError, OSError):
                    continue

        if total_bytes is None:
            try:
                total_bytes = self._get_vram_from_fdinfo(card_id)
            except Exception as e:
                logger.debug(f"Failed to get xe VRAM from fdinfo: {e}")
                return None

        if total_bytes is None or total_bytes == 0:
            return None

        total_gb = total_bytes / (1024 ** 3)

        used_gb = 0.0
        used_paths = [
            f"/sys/class/drm/card{card_id}/device/tile0/vram_used",
            f"{device_path}/tile0/vram_used",
        ]
        for used_path in used_paths:
            if os.path.exists(used_path):
                try:
                    with open(used_path, "r") as f:
                        content = f.read().strip()
                        match = re.search(r"(\d+)", content)
                        if match:
                            used_gb = int(match.group(1)) / (1024 ** 3)
                            break
                except (IOError, OSError):
                    continue

        free_gb = max(0.0, total_gb - used_gb)

        return GPUMemory(
            index=0,
            total_gb=total_gb,
            used_gb=used_gb,
            free_gb=free_gb,
            vendor=self.vendor,
            device_name=f"Intel Arc GPU (card {card_id}, xe)",
        )

    def _get_vram_from_fdinfo(self, card_id: str) -> Optional[int]:
        """Get VRAM from fdinfo for xe driver.

        The xe driver exposes memory stats via fdinfo format:
            drm-total-vram0: 23992 KiB

        Args:
            card_id: Card number string

        Returns:
            Total VRAM in bytes, or None if not found
        """
        fdinfo_path = f"/sys/kernel/debug/dri/{card_id}/clients"
        if not os.path.exists(fdinfo_path):
            return None

        try:
            with open(fdinfo_path, "r") as f:
                content = f.read()

            for line in content.split("\n"):
                if "drm-total-vram0:" in line.lower():
                    match = re.search(r"(\d+)\s*([KMGT]?i?B)", line, re.IGNORECASE)
                    if match:
                        value = int(match.group(1))
                        unit = match.group(2).upper()
                        multipliers = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024**2, "MIB": 1024**2, "GB": 1024**3, "GIB": 1024**3, "TB": 1024**4, "TIB": 1024**4}
                        return value * multipliers.get(unit, 1)
        except (IOError, OSError):
            pass

        return None

    def _query_i915_device(self, device_path: str, card_id: str) -> Optional[GPUMemory]:
        """Query VRAM for i915 driver (Arc A-series).

        Args:
            device_path: Path to /sys/class/drm/cardX/device
            card_id: Card number string

        Returns:
            GPUMemory object, or None if data incomplete
        """
        try:
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

            used_gb = 0.0
            used_path = f"{device_path}/lmem_used"
            if os.path.exists(used_path):
                with open(used_path, "r") as f:
                    used_str = f.read().strip()
                    match = re.search(r"(\d+)", used_str)
                    if match:
                        used_bytes = int(match.group(1))
                        used_gb = used_bytes / (1024 ** 3)

            free_gb = max(0.0, total_gb - used_gb)

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
