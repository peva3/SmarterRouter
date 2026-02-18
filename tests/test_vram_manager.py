"""Tests for VRAMManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from router.vram_manager import VRAMManager, VRAMExceededError


class TestVRAMManager:
    """Test VRAMManager functionality."""

    @pytest.fixture
    def vram_manager(self):
        """Create a VRAMManager instance for testing."""
        return VRAMManager(
            max_vram_gb=24.0,
            auto_unload_enabled=True,
            unload_strategy="lru",
        )

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.load_model = AsyncMock(return_value=True)
        backend.unload_model = AsyncMock(return_value=True)
        return backend

    def test_initialization(self, vram_manager):
        """Test VRAMManager initializes correctly."""
        assert vram_manager.max_vram == 24.0
        assert vram_manager.auto_unload is True
        assert vram_manager.unload_strategy == "lru"
        assert vram_manager.loaded_models == {}
        assert vram_manager.pinned_model is None

    def test_get_available_vram_empty(self, vram_manager):
        """Test available VRAM when no models loaded."""
        available = vram_manager.get_available_vram()
        expected = 24.0 - VRAMManager.FRAGMENTATION_BUFFER_GB
        assert available == expected

    def test_get_available_vram_with_models(self, vram_manager):
        """Test available VRAM with models loaded."""
        vram_manager.loaded_models = {"model1": 8.0, "model2": 4.0}
        available = vram_manager.get_available_vram()
        expected = 24.0 - VRAMManager.FRAGMENTATION_BUFFER_GB - 12.0
        assert available == expected

    def test_can_load_fits(self, vram_manager):
        """Test can_load when model fits."""
        assert vram_manager.can_load("test-model", 10.0) is True

    def test_can_load_too_large(self, vram_manager):
        """Test can_load when model is too large."""
        vram_manager.loaded_models = {"existing": 20.0}
        assert vram_manager.can_load("test-model", 5.0) is False

    def test_get_current_allocated_empty(self, vram_manager):
        """Test allocated VRAM when empty."""
        assert vram_manager.get_current_allocated() == 0.0

    def test_get_current_allocated_with_models(self, vram_manager):
        """Test allocated VRAM with models."""
        vram_manager.loaded_models = {"model1": 8.0, "model2": 4.0}
        assert vram_manager.get_current_allocated() == 12.0

    def test_get_utilization_pct_empty(self, vram_manager):
        """Test utilization percentage when empty."""
        assert vram_manager.get_utilization_pct() == 0.0

    def test_get_utilization_pct_with_models(self, vram_manager):
        """Test utilization percentage with models."""
        vram_manager.loaded_models = {"model1": 12.0}
        assert vram_manager.get_utilization_pct() == 50.0

    def test_is_loaded(self, vram_manager):
        """Test is_loaded check."""
        vram_manager.loaded_models = {"model1": 8.0}
        assert vram_manager.is_loaded("model1") is True
        assert vram_manager.is_loaded("model2") is False

    def test_get_loaded_models(self, vram_manager):
        """Test get_loaded_models returns correct list."""
        vram_manager.loaded_models = {"model1": 8.0, "model2": 4.0}
        loaded = vram_manager.get_loaded_models()
        assert "model1" in loaded
        assert "model2" in loaded

    def test_get_vram_usage_by_model(self, vram_manager):
        """Test VRAM usage by model."""
        vram_manager.loaded_models = {"model1": 8.0, "model2": 4.0}
        usage = vram_manager.get_vram_usage_by_model()
        assert usage["model1"] == 8.0
        assert usage["model2"] == 4.0

    def test_set_backend(self, vram_manager, mock_backend):
        """Test setting backend reference."""
        vram_manager.set_backend(mock_backend)
        assert vram_manager._backend_ref == mock_backend

    @pytest.mark.asyncio
    async def test_load_model_success(self, vram_manager, mock_backend):
        """Test successful model loading."""
        vram_manager.set_backend(mock_backend)
        await vram_manager.load_model("test-model", 8.0)
        
        assert "test-model" in vram_manager.loaded_models
        assert vram_manager.loaded_models["test-model"] == 8.0
        mock_backend.load_model.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, vram_manager, mock_backend):
        """Test loading an already loaded model."""
        vram_manager.set_backend(mock_backend)
        vram_manager.loaded_models = {"test-model": 8.0}
        
        await vram_manager.load_model("test-model", 8.0)
        
        mock_backend.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_pin(self, vram_manager, mock_backend):
        """Test loading and pinning a model."""
        vram_manager.set_backend(mock_backend)
        await vram_manager.load_model("test-model", 8.0, pin=True)
        
        assert vram_manager.pinned_model == "test-model"

    @pytest.mark.asyncio
    async def test_unload_model_success(self, vram_manager, mock_backend):
        """Test successful model unloading."""
        vram_manager.set_backend(mock_backend)
        vram_manager.loaded_models = {"test-model": 8.0}
        
        await vram_manager.unload_model("test-model")
        
        assert "test-model" not in vram_manager.loaded_models
        mock_backend.unload_model.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_unload_pinned_model_skipped(self, vram_manager, mock_backend):
        """Test that pinned model is not unloaded."""
        vram_manager.set_backend(mock_backend)
        vram_manager.loaded_models = {"pinned-model": 8.0}
        vram_manager.pinned_model = "pinned-model"
        
        await vram_manager.unload_model("pinned-model")
        
        assert "pinned-model" in vram_manager.loaded_models
        mock_backend.unload_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_vram_exceeded_no_auto_unload(self, mock_backend):
        """Test loading when VRAM exceeded and auto_unload disabled."""
        manager = VRAMManager(
            max_vram_gb=10.0,
            auto_unload_enabled=False,
        )
        manager.set_backend(mock_backend)
        manager.loaded_models = {"existing": 8.0}
        
        with pytest.raises(VRAMExceededError):
            await manager.load_model("new-model", 5.0)

    @pytest.mark.asyncio
    async def test_load_model_no_backend(self, vram_manager):
        """Test loading without backend set raises error."""
        with pytest.raises(RuntimeError, match="backend not set"):
            await vram_manager.load_model("test-model", 8.0)


class TestVRAMManagerStrategies:
    """Test different unloading strategies."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.load_model = AsyncMock(return_value=True)
        backend.unload_model = AsyncMock(return_value=True)
        return backend

    @pytest.mark.asyncio
    async def test_lru_strategy(self, mock_backend):
        """Test LRU unloading strategy."""
        manager = VRAMManager(
            max_vram_gb=12.0,
            auto_unload_enabled=True,
            unload_strategy="lru",
        )
        manager.set_backend(mock_backend)
        manager.loaded_models = {"model1": 4.0, "model2": 6.0}
        
        await manager.load_model("new-model", 8.0)
        
        assert "new-model" in manager.loaded_models

    @pytest.mark.asyncio
    async def test_largest_strategy(self, mock_backend):
        """Test largest-first unloading strategy."""
        manager = VRAMManager(
            max_vram_gb=12.0,
            auto_unload_enabled=True,
            unload_strategy="largest",
        )
        manager.set_backend(mock_backend)
        manager.loaded_models = {"model1": 4.0, "model2": 6.0}
        
        await manager.load_model("new-model", 8.0)
        
        assert "new-model" in manager.loaded_models
