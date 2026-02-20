"""NVIDIA GPU backend implementation using nvidia-smi."""

import logging
import re
import subprocess
from typing import List

from router.gpu_backends.base import GPUBackend, GPUMemory

logger = logging.getLogger(__name__)


class NVIDIABackend:
    """GPU backend for NVIDIA GPUs using nvidia-smi.

    This backend requires nvidia-smi to be available in the PATH.
    It works with all NVIDIA GPUs (GeForce, Quadro, Tesla, etc.)

    Detection:
        Checks if nvidia-smi command exists and returns valid output.

    Memory Query:
        Uses: nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits

    Multi-GPU Support:
        Fully supports multi-GPU systems with proper indexing.
    """

    def __init__(self, nvidia_smi_path: str = "nvidia-smi"):
        """Initialize NVIDIA backend.

        Args:
            nvidia_smi_path: Path to nvidia-smi executable (default: "nvidia-smi")
        """
        self._nvidia_smi_path = nvidia_smi_path
        self._gpu_name: str | None = None

    @property
    def vendor(self) -> str:
        return "nvidia"

    @property
    def device_name(self) -> str:
        if self._gpu_name:
            return self._gpu_name
        return "NVIDIA GPU"

    def is_available(self) -> bool:
        """Check if nvidia-smi is available and can query GPU info."""
        try:
            result = subprocess.run(
                [self._nvidia_smi_path, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse GPU name from first line
                parts = result.stdout.strip().split("\n")[0].split(",")
                if len(parts) >= 2:
                    self._gpu_name = parts[0].strip()
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
            logger.debug(f"NVIDIA backend detection failed: {e}")
        except Exception as e:
            logger.debug(f"NVIDIA backend detection error: {e}")
        return False

    def get_memory_info(self) -> List[GPUMemory]:
        """Get VRAM memory info for all NVIDIA GPUs.

        Returns:
            List of GPUMemory objects, one per detected NVIDIA GPU.

        Raises:
            ValueError: If no valid GPU data could be parsed
            subprocess.CalledProcessError: If nvidia-smi command fails
        """
        try:
            result = subprocess.run(
                [
                    self._nvidia_smi_path,
                    "--query-gpu=index,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result.check_returncode()
            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi timed out")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"nvidia-smi failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error querying NVIDIA GPUs: {e}")
            raise

    def _parse_output(self, output: str) -> List[GPUMemory]:
        """Parse nvidia-smi CSV output.

        Expected format per line:
            index, memory.total[MiB], memory.used[MiB], memory.free[MiB]

        Example:
            0, 24576 MiB, 12845 MiB, 11731 MiB
            1, 24576 MiB, 5120 MiB, 19456 MiB

        Args:
            output: Raw nvidia-smi CSV output

        Returns:
            List of GPUMemory objects

        Raises:
            ValueError: If no valid GPU data found
        """
        lines = output.strip().split("\n")
        gpus: List[GPUMemory] = []

        for line in lines:
            parts = line.split(",")
            if len(parts) < 4:
                continue

            try:
                idx = int(parts[0].strip())
                # Extract numbers from memory fields (they include "MiB" or just the number with --format=nounits)
                total_match = re.search(r"(\d+)", parts[1].strip())
                used_match = re.search(r"(\d+)", parts[2].strip())
                free_match = re.search(r"(\d+)", parts[3].strip())

                if total_match and used_match and free_match:
                    gpu_total = int(total_match.group(1))
                    gpu_used = int(used_match.group(1))
                    gpu_free = int(free_match.group(1))

                    gpus.append(
                        GPUMemory(
                            index=0,  # Will be reassigned by manager
                            total_gb=gpu_total / 1024.0,
                            used_gb=gpu_used / 1024.0,
                            free_gb=gpu_free / 1024.0,
                            vendor=self.vendor,
                            device_name=f"{self.device_name} (GPU {idx})",
                        )
                    )
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(f"Failed to parse NVIDIA GPU line: {line!r} - {e}")
                continue

        if not gpus:
            raise ValueError("No valid NVIDIA GPU data parsed from nvidia-smi output")

        return gpus
