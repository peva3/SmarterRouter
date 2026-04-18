"""Tests for dynamic model metadata system.

Covers:
- ModelMetadata dataclass and capabilities
- ModelMetadataRegistry with TTL caching
- Gemma 4 model support
- MoE and quantization-aware VRAM estimation
- Pattern-based fallback detection
"""

import pytest

from router.model_metadata import ModelMetadata, ModelMetadataRegistry


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_basic_metadata(self):
        """Create basic metadata with defaults."""
        metadata = ModelMetadata(name="llama3:8b")
        assert metadata.name == "llama3:8b"
        assert metadata.supports_vision is False
        assert metadata.supports_tool_calling is False
        assert metadata.is_moe is False
        assert metadata.parameter_count is None

    def test_vision_metadata(self):
        """Metadata with vision capability."""
        metadata = ModelMetadata(
            name="llava:7b",
            supports_vision=True,
            details_source="pattern",
            confidence=0.9,
        )
        assert metadata.supports_vision is True
        assert metadata.details_source == "pattern"
        assert metadata.confidence == 0.9

    def test_moe_metadata(self):
        """Metadata for MoE model."""
        metadata = ModelMetadata(
            name="gemma4:26b",
            is_moe=True,
            parameter_count=26_000_000_000,
            active_parameters=4_000_000_000,
        )
        assert metadata.is_moe is True
        assert metadata.parameter_count == 26_000_000_000
        assert metadata.active_parameters == 4_000_000_000


class TestGemma4Support:
    """Test Gemma 4 model support."""

    def test_gemma4_vision_detection(self):
        """Gemma 4 models are detected as vision-capable."""
        gemma4_variants = [
            "gemma4:e2b",
            "gemma4:e4b",
            "gemma4:26b",
            "gemma4:31b",
            "gemma4:latest",
        ]
        
        from router.modality import ModalityDetector
        
        for model_name in gemma4_variants:
            assert ModalityDetector._supports_vision(model_name) is True

    def test_gemma4_tool_calling_detection(self):
        """Gemma 4 models are detected as tool-calling capable."""
        gemma4_variants = [
            "gemma4:e2b",
            "gemma4:e4b",
            "gemma4:26b",
            "gemma4:31b",
        ]
        
        from router.modality import ModalityDetector
        
        for model_name in gemma4_variants:
            assert ModalityDetector._supports_tool_calling(model_name) is True

    def test_gemma4_filtering(self):
        """Gemma 4 models are included in vision filter results."""
        from router.modality import Modality, get_models_for_modality
        
        available_models = [
            "llama3:8b",
            "gemma4:e2b",
            "gemma4:26b",
            "mistral:7b",
        ]
        
        vision_models = get_models_for_modality(
            Modality.VISION,
            available_models,
            model_profiles=None,
        )
        
        assert "gemma4:e2b" in vision_models
        assert "gemma4:26b" in vision_models


class TestVRAMEstimation:
    """Test VRAM estimation for different model types."""

    def test_standard_vram_estimation(self):
        """VRAM estimation for standard dense models."""
        metadata = ModelMetadata(
            name="llama3:8b",
            parameter_count=8_000_000_000,
            is_moe=False,
        )
        vram = metadata.estimate_vram_gb()
        # 8B params * 2 bytes * 1.2 buffer = ~19.2 GB
        assert vram > 15.0
        assert vram < 25.0

    def test_moe_vram_advantage(self):
        """MoE models use less VRAM (only active params)."""
        # Standard 26B model
        standard = ModelMetadata(
            name="llama3:70b",
            parameter_count=26_000_000_000,
            is_moe=False,
        )
        
        # MoE 26B with 4B active (Gemma 4 26B)
        moe = ModelMetadata(
            name="gemma4:26b",
            parameter_count=26_000_000_000,
            is_moe=True,
            active_parameters=4_000_000_000,
        )
        
        standard_vram = standard.estimate_vram_gb()
        moe_vram = moe.estimate_vram_gb()
        
        # MoE should use significantly less VRAM
        assert moe_vram < standard_vram

    def test_quantization_vram_impact(self):
        """Quantization reduces VRAM requirements."""
        base_params = 8_000_000_000
        
        # FP16
        fp16 = ModelMetadata(
            name="llama3:8b-fp16",
            parameter_count=base_params,
            quantization="f16",
        )
        
        # Q4_K_M
        q4 = ModelMetadata(
            name="llama3:8b-q4_k_m",
            parameter_count=base_params,
            quantization="q4_k_m",
        )
        
        # Q8_0
        q8 = ModelMetadata(
            name="llama3:8b-q8_0",
            parameter_count=base_params,
            quantization="q8_0",
        )
        
        fp16_vram = fp16.estimate_vram_gb()
        q4_vram = q4.estimate_vram_gb()
        q8_vram = q8.estimate_vram_gb()
        
        # All should be positive
        assert fp16_vram > 0
        assert q4_vram > 0
        assert q8_vram > 0
        
        # Q4 should use less than FP16
        assert q4_vram < fp16_vram
        
        # Q4 should be roughly 50-60% of FP16
        assert q4_vram > fp16_vram * 0.4
        assert q4_vram < fp16_vram * 0.7
        
        # Q8 should be between Q4 and FP16
        assert q4_vram < q8_vram < fp16_vram

    def test_unknown_model_default(self):
        """Unknown models get reasonable default VRAM estimate."""
        metadata = ModelMetadata(name="unknown-model")
        vram = metadata.estimate_vram_gb()
        # Default should be reasonable (not too small, not too large)
        assert vram > 4.0
        assert vram < 64.0


class TestMetadataRegistry:
    """Test ModelMetadataRegistry functionality."""

    @pytest.mark.asyncio
    async def test_registry_creation(self):
        """Can create and use registry."""
        registry = ModelMetadataRegistry(ttl_seconds=60)
        metadata = await registry.get_metadata("llama3:8b")
        assert metadata.name == "llama3:8b"

    @pytest.mark.asyncio  
    async def test_registry_caching(self):
        """Registry caches metadata lookups."""
        registry = ModelMetadataRegistry(ttl_seconds=300)
        
        # First lookup
        metadata1 = await registry.get_metadata("test-model")
        
        # Second lookup should use cache
        metadata2 = await registry.get_metadata("test-model")
        
        # Should be same object (cached)
        assert metadata1 is metadata2

def test_pattern_fallback(self):
    """Pattern-based fallback detection works."""
    from router.modality import _supports_vision, _supports_tool_calling

    # Should detect vision from pattern
    assert _supports_vision("llava-34b") is True
    assert _supports_vision("gpt-4o") is True

    # Should detect tool calling from pattern
    assert _supports_tool_calling("gpt-4") is True
    assert _supports_tool_calling("qwen2.5") is True
