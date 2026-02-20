"""AMD GPU backend implementation using rocm-smi or sysfs."""

import logging
import os
import re
import subprocess
from typing import List, Optional

from router.gpu_backends.base import GPUBackend, GPUMemory

logger = logging.getLogger(__name__)

# AMD vendor ID for sysfs detection
AMD_VENDOR_ID = "0x1002"


class AMDBackend:
    """GPU backend for AMD GPUs using rocm-smi or sysfs fallback.

    This backend supports both ROCm and older AMD GPUs via sysfs.

    Detection Strategy:
        1. Try rocm-smi (preferred, provides detailed info)
        2. Fall back to sysfs (/sys/class/drm/card*/device/)

    Memory Query:
        Primary: rocm-smi --showmeminfo vram
        Fallback: /sys/class/drm/card*/device/mem_info_vram_* (total, used, etc.)

    Supported GPUs:
        - AMD Radeon Instinct (MI series)
        - AMD Radeon Pro
        - AMD Radeon RX (with ROCm support)
        - Older AMD GPUs via sysfs (limited info)

    Note:
        VRAM reporting quality depends on driver and GPU model.
        Some older GPUs may only provide total memory, not used/free.
    """

    def __init__(self, rocm_smi_path: str = "rocm-smi"):
        """Initialize AMD backend.

        Args:
            rocm_smi_path: Path to rocm-smi executable (default: "rocm-smi")
        """
        self._rocm_smi_path = rocm_smi_path
        self._sysfs_paths: List[str] = []
        self._detect_sysfs_cards()

    def _detect_sysfs_cards(self):
        """Detect AMD DRM cards in sysfs at init."""
        try:
            for card in os.listdir("/sys/class/drm"):
                if card.startswith("card"):
                    device_path = f"/sys/class/drm/{card}/device"
                    vendor_path = f"{device_path}/vendor"
                    if os.path.exists(vendor_path):
                        try:
                            with open(vendor_path, "r") as f:
                                vendor = f.read().strip()
                                if vendor == AMD_VENDOR_ID:
                                    self._sysfs_paths.append(device_path)
                        except (IOError, OSError):
                            continue
        except (FileNotFoundError, OSError):
            pass

    @property
    def vendor(self) -> str:
        return "amd"

    @property
    def device_name(self) -> str:
        # Return generic AMD GPU name; will be refined from rocm-smi if available
        return "AMD GPU"

    def is_available(self) -> bool:
        """Check if AMD GPUs are available via rocm-smi or sysfs."""
        # Try rocm-smi first
        try:
            result = subprocess.run(
                [self._rocm_smi_path, "--showid"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass
        except Exception as e:
            logger.debug(f"rocm-smi check failed: {e}")

        # Fallback to sysfs detection
        return len(self._sysfs_paths) > 0

    def get_memory_info(self) -> List[GPUMemory]:
        """Get VRAM memory info for all AMD GPUs.

        Tries rocm-smi first, falls back to sysfs if rocm-smi fails.

        Returns:
            List of GPUMemory objects, one per detected AMD GPU.

        Raises:
            ValueError: If no GPU data could be obtained
        """
        # Try rocm-smi method
        try:
            return self._query_rocm_smi()
        except Exception as e:
            logger.debug(f"rocm-smi query failed, trying sysfs: {e}")

        # Fallback to sysfs
        try:
            return self._query_sysfs()
        except Exception as e:
            logger.error(f"Both rocm-smi and sysfs failed for AMD: {e}")
            raise ValueError(f"Unable to query AMD GPU memory: {e}")

    def _query_rocm_smi(self) -> List[GPUMemory]:
        """Query AMD GPUs using rocm-smi command."""
        try:
            result = subprocess.run(
                [self._rocm_smi_path, "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result.check_returncode()

            import json
            data = json.loads(result.stdout)
            gpus: List[GPUMemory] = []

            # rocm-smi JSON format varies by version
            # Expected: {"system": {"gpu": [{"GPU": <id>, "VRAM Total": <MiB>, ...}]}}
            gpu_list = data.get("system", {}).get("gpu", [])
            if not gpu_list and isinstance(data, list):
                # Some versions output direct array
                gpu_list = data

            for idx, gpu_data in enumerate(gpu_list):
                # Extract VRAM total (in MiB)
                total_mib = self._extract_rocm_memory(gpu_data, "VRAM Total")
                used_mib = self._extract_rocm_memory(gpu_data, "VRAM Used")
                # Calculate free if not provided
                if used_mib is not None and total_mib:
                    free_mib = total_mib - used_mib
                else:
                    free_mib = None

                if total_mib:
                    gpu_name = gpu_data.get("GPU", f"AMD GPU {idx}")
                    gpus.append(
                        GPUMemory(
                            index=0,
                            total_gb=total_mib / 1024.0,
                            used_gb=used_mib / 1024.0 if used_mib else 0.0,
                            free_gb=free_mib / 1024.0 if free_mib else total_mib / 1024.0,
                            vendor=self.vendor,
                            device_name=f"AMD {gpu_name}",
                        )
                    )

            if not gpus:
                raise ValueError("No valid GPU data from rocm-smi JSON")

            return gpus

        except json.JSONDecodeError as e:
            logger.warning(f"rocm-smi JSON parsing failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"rocm-smi query error: {e}")
            raise

    def _extract_rocm_memory(self, gpu_data: dict, key: str) -> Optional[int]:
        """Extract memory value in MiB from rocm-smi data.

        Handles formats like "24576 MiB", "24576", etc.
        """
        value = gpu_data.get(key, 0)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"(\d+)", value)
            if match:
                return int(match.group(1))
        return None

    def _query_sysfs(self) -> List[GPUMemory]:
        """Query AMD GPUs using sysfs entries.

        Sysfs paths for AMD GPUs:
            /sys/class/drm/cardX/device/mem_info_vram_total
            /sys/class/drm/cardX/device/mem_info_vram_used
            /sys/class/drm/cardX/device/mem_info_vram_free
            /sys/class/drm/cardX/device/mem_info_vram_pmiss (if used)

        Note: Not all AMD GPUs/drivers expose used/free, only total.
        """
        gpus: List[GPUMemory] = []

        for device_path in self._sysfs_paths:
            try:
                # Read total VRAM
                total_path = f"{device_path}/mem_info_vram_total"
                if not os.path.exists(total_path):
                    continue

                with open(total_path, "r") as f:
                    total_str = f.read().strip()
                    total_match = re.search(r"(\d+)", total_str)
                    if not total_match:
                        continue
                    total_bytes = int(total_match.group(1))
                    total_gb = total_bytes / (1024 ** 3)

                # Try to get used VRAM
                used_gb = 0.0
                free_gb = total_gb
                for used_key in ["mem_info_vram_used", "mem_info_vram_pmiss"]:
                    used_path = f"{device_path}/{used_key}"
                    if os.path.exists(used_path):
                        with open(used_path, "r") as f:
                            used_str = f.read().strip()
                            used_match = re.search(r"(\d+)", used_str)
                            if used_match:
                                used_bytes = int(used_match.group(1))
                                used_gb = used_bytes / (1024 ** 3)
                                free_gb = total_gb - used_gb
                                break

                # Get card number for device name
                card_match = re.search(r"card(\d+)", device_path)
                card_id = card_match.group(1) if card_match else "?"

                gpus.append(
                    GPUMemory(
                        index=0,
                        total_gb=total_gb,
                        used_gb=used_gb,
                        free_gb=free_gb,
                        vendor=self.vendor,
                        device_name=f"AMD GPU (card {card_id})",
                    )
                )
            except (IOError, OSError, ValueError) as e:
                logger.debug(f"Failed to read AMD GPU sysfs at {device_path}: {e}")
                continue

        if not gpus:
            raise ValueError("No AMD GPU data obtained from sysfs")

        return gpus
