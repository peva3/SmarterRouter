"""Tests for VRAMMonitor."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from router.vram_monitor import VRAMMonitor, VRAMMetrics, GPUMemory


class TestGPUMemory:
    """Test GPUMemory dataclass."""

    def test_gpu_memory_creation(self):
        """Test GPUMemory is created correctly."""
        gpu = GPUMemory(
            index=0,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
        )
        assert gpu.index == 0
        assert gpu.total_gb == 24.0
        assert gpu.used_gb == 12.0
        assert gpu.free_gb == 12.0


class TestVRAMMetrics:
    """Test VRAMMetrics dataclass."""

    def test_vram_metrics_creation(self):
        """Test VRAMMetrics is created correctly."""
        metrics = VRAMMetrics(
            timestamp=12345.0,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
            utilization_pct=50.0,
            models_loaded=["model1"],
            per_model_vram_gb={"model1": 8.0},
            gpus=[GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0)],
        )
        assert metrics.timestamp == 12345.0
        assert metrics.total_gb == 24.0
        assert metrics.utilization_pct == 50.0

    def test_to_log_string_basic(self):
        """Test basic log string format."""
        metrics = VRAMMetrics(
            timestamp=12345.0,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
            utilization_pct=50.0,
            models_loaded=[],
            per_model_vram_gb={},
            gpus=[GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0)],
        )
        log_str = metrics.to_log_string()
        assert "VRAM:" in log_str
        assert "12.0/24.0GB" in log_str
        assert "50.0%" in log_str

    def test_to_log_string_with_models(self):
        """Test log string with loaded models."""
        metrics = VRAMMetrics(
            timestamp=12345.0,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
            utilization_pct=50.0,
            models_loaded=["llama3"],
            per_model_vram_gb={"llama3": 8.0},
            gpus=[GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0)],
        )
        log_str = metrics.to_log_string()
        assert "llama3:8.0GB" in log_str

    def test_to_log_string_multi_gpu(self):
        """Test log string with multiple GPUs."""
        metrics = VRAMMetrics(
            timestamp=12345.0,
            total_gb=48.0,
            used_gb=24.0,
            free_gb=24.0,
            utilization_pct=50.0,
            models_loaded=[],
            per_model_vram_gb={},
            gpus=[
                GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0),
                GPUMemory(index=1, total_gb=24.0, used_gb=12.0, free_gb=12.0),
            ],
        )
        log_str = metrics.to_log_string()
        assert "GPU0:" in log_str
        assert "GPU1:" in log_str


class TestVRAMMonitor:
    """Test VRAMMonitor functionality."""

    @pytest.fixture
    def monitor(self):
        """Create a VRAMMonitor instance."""
        return VRAMMonitor(
            interval=30,
            total_vram_gb=24.0,
            log_interval=60,
        )

    def test_initialization(self, monitor):
        """Test VRAMMonitor initializes correctly."""
        assert monitor.interval == 30
        assert monitor.total_vram_gb == 24.0
        assert monitor.log_interval == 60
        assert monitor._running is False
        assert monitor._samples == []

    def test_get_current_empty(self, monitor):
        """Test get_current returns None when no samples."""
        assert monitor.get_current() is None

    def test_get_current_with_sample(self, monitor):
        """Test get_current returns latest sample."""
        sample = VRAMMetrics(
            timestamp=12345.0,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
            utilization_pct=50.0,
            models_loaded=[],
            per_model_vram_gb={},
            gpus=[GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0)],
        )
        monitor._samples.append(sample)
        
        result = monitor.get_current()
        assert result == sample

    def test_get_history_empty(self, monitor):
        """Test get_history returns empty list when no samples."""
        assert monitor.get_history(minutes=10) == []

    def test_get_history_with_samples(self, monitor):
        """Test get_history filters by time."""
        import time
        
        now = time.time()
        recent = VRAMMetrics(
            timestamp=now - 60,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
            utilization_pct=50.0,
            models_loaded=[],
            per_model_vram_gb={},
            gpus=[GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0)],
        )
        old = VRAMMetrics(
            timestamp=now - 600,
            total_gb=24.0,
            used_gb=12.0,
            free_gb=12.0,
            utilization_pct=50.0,
            models_loaded=[],
            per_model_vram_gb={},
            gpus=[GPUMemory(index=0, total_gb=24.0, used_gb=12.0, free_gb=12.0)],
        )
        
        monitor._samples = [old, recent]
        
        history = monitor.get_history(minutes=5)
        assert len(history) == 1
        assert history[0] == recent

    @pytest.mark.asyncio
    async def test_start_without_nvidia(self, monitor):
        """Test start when nvidia-smi is not available."""
        monitor.has_nvidia = False
        
        await monitor.start()
        
        assert monitor._task is None
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_stop(self, monitor):
        """Test stopping the monitor."""
        monitor._running = True
        
        await monitor.stop()
        
        assert monitor._running is False

    def test_parse_output_single_gpu(self, monitor):
        """Test parsing nvidia-smi output for single GPU."""
        output = "0, 24576 MiB, 12845 MiB, 11731 MiB"
        
        total_mb, used_mb, free_mb, gpus = monitor._parse_output(output)
        
        assert total_mb == 24576
        assert used_mb == 12845
        assert free_mb == 11731
        assert len(gpus) == 1
        assert gpus[0].index == 0

    def test_parse_output_multi_gpu(self, monitor):
        """Test parsing nvidia-smi output for multiple GPUs."""
        output = """0, 24576 MiB, 12845 MiB, 11731 MiB
1, 24576 MiB, 10000 MiB, 14576 MiB"""
        
        total_mb, used_mb, free_mb, gpus = monitor._parse_output(output)
        
        assert total_mb == 49152
        assert used_mb == 22845
        assert len(gpus) == 2

    def test_parse_output_invalid(self, monitor):
        """Test parsing invalid output raises error."""
        output = "invalid data"
        
        with pytest.raises(ValueError):
            monitor._parse_output(output)

    @patch('subprocess.run')
    def test_check_nvidia_smi_available(self, mock_run, monitor):
        """Test nvidia-smi detection when available."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3090, 24576 MiB"
        )
        
        result = monitor._check_nvidia_smi()
        
        assert result is True
        assert monitor.gpu_name == "NVIDIA GeForce RTX 3090"

    @patch('subprocess.run')
    def test_check_nvidia_smi_not_available(self, mock_run, monitor):
        """Test nvidia-smi detection when not available."""
        mock_run.side_effect = FileNotFoundError()
        
        result = monitor._check_nvidia_smi()
        
        assert result is False


class TestVRAMMonitorAppState:
    """Test VRAMMonitor with app_state integration."""

    def test_sample_with_vram_manager(self):
        """Test sampling includes VRAM manager model info."""
        mock_app_state = MagicMock()
        mock_vram_manager = MagicMock()
        mock_vram_manager.loaded_models = {"llama3": 8.0}
        mock_app_state.vram_manager = mock_vram_manager
        
        monitor = VRAMMonitor(
            interval=30,
            total_vram_gb=24.0,
            app_state=mock_app_state,
        )
        
        monitor.has_nvidia = True
        monitor._run_nvidia_smi = lambda: "0, 24576 MiB, 12845 MiB, 11731 MiB"
        
        import asyncio
        metrics = asyncio.get_event_loop().run_until_complete(monitor._sample())
        
        assert "llama3" in metrics.models_loaded
        assert metrics.per_model_vram_gb["llama3"] == 8.0
