"""
Tests for router/entrypoint.py to prevent nested event loop issues.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent directory to path to import router modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.entrypoint import main, start_server


class TestEntrypointEventLoop:
    """Tests to ensure entrypoint doesn't create nested event loops."""

    @pytest.mark.asyncio
    async def test_start_server_does_not_call_uvicorn_run(self):
        """Test that start_server() uses uvicorn.Server programmatically."""
        mock_server = AsyncMock()
        mock_config = MagicMock()

        # Mock uvicorn module since it's imported inside start_server
        with patch("router.entrypoint.uvicorn") as mock_uvicorn:
            mock_uvicorn.Config.return_value = mock_config
            mock_uvicorn.Server.return_value = mock_server

            await start_server()

            # Verify uvicorn.Config was called
            mock_uvicorn.Config.assert_called_once()

            # Verify uvicorn.Server was created with config
            mock_uvicorn.Server.assert_called_once_with(mock_config)

            # Verify server.serve() was awaited
            mock_server.serve.assert_awaited_once()

            # Verify uvicorn.run() was NOT called
            assert not mock_uvicorn.run.called

    @pytest.mark.asyncio
    async def test_main_calls_async_start_server(self):
        """Test that main() awaits start_server()."""
        with (
            patch("router.entrypoint.auto_setup", new_callable=AsyncMock) as mock_auto_setup,
            patch(
                "router.entrypoint.validate_environment", new_callable=AsyncMock
            ) as mock_validate,
            patch("router.entrypoint.start_server", new_callable=AsyncMock) as mock_start_server,
            patch("router.entrypoint.setup_logging"),
        ):
            mock_auto_setup.return_value = False
            mock_validate.return_value = {
                "config_file_exists": True,
                "config_valid": True,
                "ollama_reachable": True,
                "models_available": True,
                "gpu_detected": True,
                "issues": [],
            }

            # Run main
            await main()

            # Verify start_server was awaited (not called synchronously)
            mock_start_server.assert_awaited_once()

    def test_no_nested_asyncio_run_in_entrypoint(self):
        """Test that the entrypoint script doesn't have nested asyncio.run calls."""
        with open("router/entrypoint.py") as f:
            content = f.read()

        # Count occurrences of asyncio.run
        run_count = content.count("asyncio.run")

        # There should be exactly one asyncio.run at the module level
        assert run_count == 1, f"Found {run_count} asyncio.run() calls, expected exactly 1"

        # Verify it's at the bottom in the __name__ == "__main__" block
        lines = content.split("\n")
        run_lines = [i for i, line in enumerate(lines) if "asyncio.run" in line]

        assert len(run_lines) == 1, "Should have exactly one asyncio.run line"

        # Check it's in the main guard
        main_guard_start = None
        for i, line in enumerate(lines):
            if '__name__ == "__main__"' in line:
                main_guard_start = i
                break

        assert main_guard_start is not None, "Should have __name__ == '__main__' guard"

        # The asyncio.run should be after the main guard
        assert run_lines[0] > main_guard_start, "asyncio.run should be inside __main__ guard"

    @pytest.mark.asyncio
    async def test_entrypoint_can_be_imported_without_running(self):
        """Test that importing entrypoint doesn't immediately run code."""
        # This test ensures no code runs on import (except top-level definitions)
        import router.entrypoint

        # Just importing should work
        assert hasattr(router.entrypoint, "main")
        assert hasattr(router.entrypoint, "start_server")

        # The module should have been imported successfully
        assert router.entrypoint.__name__ == "router.entrypoint"
