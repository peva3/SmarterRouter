"""AMD GPU backend implementation using rocm-smi or sysfs.

Supports both discrete AMD GPUs and integrated APUs (Accelerated Processing Units)
with unified memory architecture.

Memory Detection Strategy:
    1. Try rocm-smi (preferred, provides detailed info)
    2. Fall back to sysfs:
       - Discrete GPUs: mem_info_vram_* (dedicated VRAM)
       - APUs: mem_info_gtt_* (unified memory pool via GTT)

APU vs Discrete Detection:
    APUs typically have:
    - Very small or zero VRAM (BIOS carve-out, usually 512MB-8GB)
    - Large GTT pool (shared system memory)
    - Device names like "Radeon Graphics", "Vega", "Graphics"
    
    Discrete GPUs have:
    - Large dedicated VRAM
    - GTT may exist but VRAM is primary
"""

import json
import logging
import os
import re
import subprocess
from typing import List, Optional

from router.gpu_backends.base import GPUBackend, GPUMemory

logger = logging.getLogger(__name__)

AMD_VENDOR_ID = "0x1002"

VRAM_CUTOFF_GB = 4.0


class AMDBackend:
    """GPU backend for AMD GPUs using rocm-smi or sysfs fallback.

    This backend supports:
    - Discrete AMD GPUs (Radeon RX series, Radeon Pro, Instinct)
    - Integrated AMD APUs (Radeon Graphics, Ryzen mobile GPUs)
    
    For APUs with unified memory, uses GTT (Graphics Translation Table)
    to report the full shared memory pool accessible to the GPU.
    """

    def __init__(
        self,
        rocm_smi_path: str = "rocm-smi",
        unified_memory_gb: Optional[float] = None,
    ):
        """Initialize AMD backend.

        Args:
            rocm_smi_path: Path to rocm-smi executable (default: "rocm-smi")
            unified_memory_gb: For APUs with unified memory - override auto-detected
                             value. Useful when sysfs reports incorrect GTT size.
        """
        self._rocm_smi_path = rocm_smi_path
        self._unified_memory_gb = unified_memory_gb
        self._sysfs_paths: List[str] = []
        self._is_apu: bool = False
        self._detected_device_name: str = "AMD GPU"
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
        return self._detected_device_name

    def is_available(self) -> bool:
        """Check if AMD GPUs are available via rocm-smi or sysfs."""
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

        return len(self._sysfs_paths) > 0

    def get_memory_info(self) -> List[GPUMemory]:
        """Get memory info for all AMD GPUs.

        Tries rocm-smi first, falls back to sysfs if rocm-smi fails.
        For APUs, prioritizes GTT (unified memory) over VRAM.

        Returns:
            List of GPUMemory objects, one per detected AMD GPU.

        Raises:
            ValueError: If no GPU data could be obtained
        """
        if self._unified_memory_gb is not None:
            try:
                return self._query_sysfs()
            except Exception as e:
                logger.error(f"sysfs query failed with manual override: {e}")
                raise ValueError(f"Unable to query AMD GPU memory with override: {e}")

        try:
            gpus = self._query_rocm_smi()
            for gpu in gpus:
                if gpu.total_gb < VRAM_CUTOFF_GB:
                    logger.info(
                        f"AMD GPU {gpu.device_name} appears to be APU (VRAM={gpu.total_gb:.1f}GB < {VRAM_CUTOFF_GB}GB cutoff), "
                        "falling back to sysfs for GTT unified memory"
                    )
                    raise ValueError("APU detected, use sysfs for GTT")
            return gpus
        except Exception as e:
            logger.debug(f"rocm-smi query failed or APU detected, trying sysfs: {e}")

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

            data = json.loads(result.stdout)
            gpus: List[GPUMemory] = []

            gpu_list = data.get("system", {}).get("gpu", [])
            if not gpu_list and isinstance(data, list):
                gpu_list = data

            for idx, gpu_data in enumerate(gpu_list):
                total_mib = self._extract_rocm_memory(gpu_data, "VRAM Total")
                used_mib = self._extract_rocm_memory(gpu_data, "VRAM Used")
                
                if used_mib is not None and total_mib:
                    free_mib = total_mib - used_mib
                else:
                    free_mib = None

                if total_mib:
                    gpu_name = gpu_data.get("GPU", f"AMD GPU {idx}")
                    total_gb = total_mib / 1024.0
                    
                    self._is_apu = total_gb < VRAM_CUTOFF_GB
                    self._detected_device_name = f"AMD {gpu_name}"
                    
                    gpus.append(
                        GPUMemory(
                            index=0,
                            total_gb=total_gb,
                            used_gb=used_mib / 1024.0 if used_mib else 0.0,
                            free_gb=free_mib / 1024.0 if free_mib else total_gb,
                            vendor=self.vendor,
                            device_name=self._detected_device_name,
                        )
                    )

            if not gpus:
                raise ValueError("No valid GPU data from rocm-smi JSON")

            return gpus

        except json.JSONDecodeError as e:
            logger.debug(f"rocm-smi JSON parsing failed: {e}")
            raise
        except Exception as e:
            logger.debug(f"rocm-smi query error: {e}")
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

        For discrete GPUs: Uses mem_info_vram_* for dedicated VRAM.
        For APUs: Uses mem_info_gtt_* for unified memory pool.
        
        GTT (Graphics Translation Table) represents the shared system memory
        accessible to APU GPUs. This is the actual usable memory for APUs,
        not the small BIOS VRAM carve-out.
        """
        gpus: List[GPUMemory] = []

        for device_path in self._sysfs_paths:
            try:
                gpu_info = self._query_single_device(device_path)
                if gpu_info:
                    gpus.append(gpu_info)
            except (IOError, OSError, ValueError) as e:
                logger.debug(f"Failed to read AMD GPU sysfs at {device_path}: {e}")
                continue

        if not gpus:
            raise ValueError("No AMD GPU data obtained from sysfs")

        return gpus

    def _query_single_device(self, device_path: str) -> Optional[GPUMemory]:
        """Query a single AMD GPU device.

        Returns None if the device cannot be read.
        """
        vram_total_path = f"{device_path}/mem_info_vram_total"
        gtt_total_path = f"{device_path}/mem_info_gtt_total"
        
        vram_total_bytes = self._read_sysfs_memory(vram_total_path)
        gtt_total_bytes = self._read_sysfs_memory(gtt_total_path)
        
        if vram_total_bytes is None and gtt_total_bytes is None:
            return None

        vram_total_gb = vram_total_bytes / (1024 ** 3) if vram_total_bytes else 0.0
        gtt_total_gb = gtt_total_bytes / (1024 ** 3) if gtt_total_bytes else 0.0

        self._is_apu = vram_total_gb < VRAM_CUTOFF_GB
        
        card_match = re.search(r"card(\d+)", device_path)
        card_id = card_match.group(1) if card_match else "?"

        device_model = self._read_device_model(device_path)
        
        if self._unified_memory_gb is not None:
            total_gb = self._unified_memory_gb
            self._detected_device_name = f"AMD APU (card {card_id}, {device_model}) - Manual Override"
            used_gb = 0.0
            free_gb = total_gb
            logger.info(
                f"AMD APU detected with manual override: {self._detected_device_name}, "
                f"total={total_gb:.1f}GB (manual override)"
            )
        elif self._is_apu and gtt_total_gb > 0:
            total_gb = gtt_total_gb
            self._detected_device_name = f"AMD APU (card {card_id}, {device_model})"
            
            gtt_used_bytes = self._read_sysfs_memory(f"{device_path}/mem_info_gtt_used")
            used_gb = gtt_used_bytes / (1024 ** 3) if gtt_used_bytes else 0.0
            free_gb = total_gb - used_gb
            
            logger.info(
                f"AMD APU detected: {self._detected_device_name}, "
                f"VRAM={vram_total_gb:.1f}GB, GTT(unified)={gtt_total_gb:.1f}GB"
            )
        else:
            total_gb = vram_total_gb
            self._detected_device_name = f"AMD GPU (card {card_id}, {device_model})"
            
            used_gb = 0.0
            for used_key in ["mem_info_vram_used", "mem_info_vram_pmiss"]:
                used_bytes = self._read_sysfs_memory(f"{device_path}/{used_key}")
                if used_bytes:
                    used_gb = used_bytes / (1024 ** 3)
                    break
            
            free_gb = total_gb - used_gb
            logger.info(
                f"AMD discrete GPU detected: {self._detected_device_name}, "
                f"VRAM={total_gb:.1f}GB"
            )

        return GPUMemory(
            index=0,
            total_gb=total_gb,
            used_gb=used_gb,
            free_gb=free_gb,
            vendor=self.vendor,
            device_name=self._detected_device_name,
        )

    def _read_sysfs_memory(self, path: str) -> Optional[int]:
        """Read a memory value from sysfs path.

        Returns bytes as integer, or None if path doesn't exist or can't be read.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                match = re.search(r"(\d+)", content)
                if match:
                    return int(match.group(1))
        except (IOError, OSError):
            pass
        return None

    def _read_device_model(self, device_path: str) -> str:
        """Try to read device model name from sysfs."""
        model_paths = [
            f"{device_path}/product_name",
            f"{device_path}/device/product_name",
            f"{device_path}/gpu_id",
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        model = f.read().strip()
                        if model and model != "AMD":
                            return model[:30]
                except (IOError, OSError):
                    continue
        
        return "Radeon Graphics"
