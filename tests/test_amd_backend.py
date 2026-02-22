"""Tests for AMD GPU backend, including APU detection."""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from router.gpu_backends.amdgpu import AMDBackend, VRAM_CUTOFF_GB
from router.gpu_backends.base import GPUMemory


class TestAMDBackendDetection:
    """Test AMD GPU detection logic."""

    def test_is_available_no_rocm_no_sysfs(self):
        """Test is_available returns False when no rocm-smi or sysfs."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = []
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        with patch('subprocess.run', side_effect=FileNotFoundError()):
            result = backend.is_available()
            assert result is False

    def test_is_available_with_sysfs(self):
        """Test is_available returns True when sysfs paths exist."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = ["/sys/class/drm/card0/device"]
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        with patch('subprocess.run', side_effect=FileNotFoundError()):
            result = backend.is_available()
            assert result is True


class TestAMDBackendAPUDetection:
    """Test AMD APU vs discrete GPU detection."""

    def test_apu_detection_small_vram(self):
        """Test APU is detected when VRAM is below cutoff."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = []
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        vram_total_gb = 2.0
        assert vram_total_gb < VRAM_CUTOFF_GB
        backend._is_apu = vram_total_gb < VRAM_CUTOFF_GB

        assert backend._is_apu is True

    def test_discrete_detection_large_vram(self):
        """Test discrete GPU is detected when VRAM is above cutoff."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = []
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        vram_total_gb = 16.0
        assert vram_total_gb >= VRAM_CUTOFF_GB
        backend._is_apu = vram_total_gb < VRAM_CUTOFF_GB

        assert backend._is_apu is False

    def test_vram_cutoff_value(self):
        """Test VRAM cutoff is 4GB."""
        assert VRAM_CUTOFF_GB == 4.0


class TestAMDBackendMemoryQuery:
    """Test AMD GPU memory querying."""

    def test_query_sysfs_discrete_gpu(self):
        """Test sysfs query for discrete GPU uses VRAM."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = ["/sys/class/drm/card0/device"]
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        def mock_read_sysfs(path):
            if "vram_total" in path:
                return 16 * 1024 * 1024 * 1024
            elif "vram_used" in path:
                return 4 * 1024 * 1024 * 1024
            elif "gtt_total" in path:
                return 32 * 1024 * 1024 * 1024
            return None

        backend._read_sysfs_memory = mock_read_sysfs
        backend._read_device_model = lambda x: "Radeon RX 7900 XT"

        result = backend._query_single_device("/sys/class/drm/card0/device")

        assert result is not None
        assert result.total_gb == 16.0
        assert result.used_gb == 4.0
        assert result.vendor == "amd"

    def test_query_sysfs_apu_uses_gtt(self):
        """Test sysfs query for APU uses GTT pool."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = ["/sys/class/drm/card0/device"]
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        def mock_read_sysfs(path):
            if "vram_total" in path:
                return 512 * 1024 * 1024
            elif "gtt_total" in path:
                return 58 * 1024 * 1024 * 1024
            elif "gtt_used" in path:
                return 2 * 1024 * 1024 * 1024
            return None

        backend._read_sysfs_memory = mock_read_sysfs
        backend._read_device_model = lambda x: "Radeon 890M"

        result = backend._query_single_device("/sys/class/drm/card0/device")

        assert result is not None
        assert result.total_gb == 58.0
        assert result.used_gb == 2.0
        assert result.vendor == "amd"
        assert "APU" in result.device_name

    def test_get_memory_info_apu_falls_back_to_sysfs(self):
        """Test get_memory_info falls back to sysfs for APUs."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = ["/sys/class/drm/card0/device"]
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        def mock_read_sysfs(path):
            if "vram_total" in path:
                return 2 * 1024 * 1024 * 1024
            elif "gtt_total" in path:
                return 58 * 1024 * 1024 * 1024
            elif "gtt_used" in path:
                return 0
            return None

        backend._read_sysfs_memory = mock_read_sysfs
        backend._read_device_model = lambda x: "Radeon 890M"

        result = backend.get_memory_info()

        assert len(result) == 1
        assert result[0].total_gb == 58.0
        assert "APU" in result[0].device_name


class TestAMDBackendManualOverride:
    """Test manual unified memory override."""

    def test_manual_override_takes_precedence(self):
        """Test that manual override overrides auto-detection."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = 58.0
        backend._sysfs_paths = ["/sys/class/drm/card0/device"]
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        def mock_read_sysfs(path):
            if "vram_total" in path:
                return 7 * 1024 * 1024 * 1024
            elif "gtt_total" in path:
                return 60 * 1024 * 1024 * 1024
            return None

        backend._read_sysfs_memory = mock_read_sysfs
        backend._read_device_model = lambda x: "Radeon 890M"

        result = backend._query_single_device("/sys/class/drm/card0/device")

        assert result is not None
        assert result.total_gb == 58.0
        assert "Manual Override" in result.device_name

    def test_manual_override_none_uses_autodetect(self):
        """Test that None override uses auto-detection."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"
        backend._unified_memory_gb = None
        backend._sysfs_paths = ["/sys/class/drm/card0/device"]
        backend._is_apu = False
        backend._detected_device_name = "AMD GPU"

        def mock_read_sysfs(path):
            if "vram_total" in path:
                return 2 * 1024 * 1024 * 1024
            elif "gtt_total" in path:
                return 58 * 1024 * 1024 * 1024
            elif "gtt_used" in path:
                return 0
            return None

        backend._read_sysfs_memory = mock_read_sysfs
        backend._read_device_model = lambda x: "Radeon 890M"

        result = backend._query_single_device("/sys/class/drm/card0/device")

        assert result is not None
        assert result.total_gb == 58.0
        assert "Manual Override" not in result.device_name


class TestAMDBackendProperties:
    """Test AMD backend properties."""

    def test_vendor_property(self):
        """Test vendor property returns 'amd'."""
        backend = AMDBackend.__new__(AMDBackend)
        assert backend.vendor == "amd"

    def test_device_name_property(self):
        """Test device_name property returns detected name."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._detected_device_name = "AMD Radeon RX 7900 XT"
        assert backend.device_name == "AMD Radeon RX 7900 XT"


class TestAMDBackendExtractRocmMemory:
    """Test rocm-smi memory extraction."""

    def test_extract_integer_value(self):
        """Test extraction of integer memory value."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"

        result = backend._extract_rocm_memory({"VRAM Total": 24576}, "VRAM Total")
        assert result == 24576

    def test_extract_string_with_mib(self):
        """Test extraction of string with MiB suffix."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"

        result = backend._extract_rocm_memory({"VRAM Total": "24576 MiB"}, "VRAM Total")
        assert result == 24576

    def test_extract_string_without_suffix(self):
        """Test extraction of string without suffix."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"

        result = backend._extract_rocm_memory({"VRAM Total": "24576"}, "VRAM Total")
        assert result == 24576

    def test_extract_missing_key(self):
        """Test extraction returns 0 for missing key (default value)."""
        backend = AMDBackend.__new__(AMDBackend)
        backend._rocm_smi_path = "rocm-smi"

        result = backend._extract_rocm_memory({}, "VRAM Total")
        assert result == 0
