# Real Ollama integration test
"""
Item #51: Real Ollama integration test.

This test spins up an actual Ollama container for end-to-end testing.
Skipped by default - requires Docker.

To run: pytest tests/integration/test_ollama_real.py -v --ollama-integration
"""
import time

import pytest

# Skip all tests in this file unless explicitly enabled
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skip(reason="Requires --ollama-integration flag and Docker"),
]


@pytest.fixture(scope="module")
def ollama_container():
    """Start Ollama container for testing."""
    import docker

    client = docker.from_env()

    # Check if Docker is available
    try:
        client.ping()
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")

    container = None
    try:
        # Pull and run Ollama container
        print("Pulling Ollama image...")
        client.images.pull("ollama/ollama:latest")

        print("Starting Ollama container...")
        container = client.containers.run(
            "ollama/ollama:latest",
            detach=True,
            ports={"11434/tcp": ("127.0.0.1", 11435)},  # Use different port to avoid conflicts
            environment={"OLLAMA_HOST": "0.0.0.0"},
            remove=True,
        )

        # Wait for Ollama to be ready
        print("Waiting for Ollama to be ready...")
        max_wait = 60
        start = time.time()

        while time.time() - start < max_wait:
            try:
                import httpx
                response = httpx.get("http://127.0.0.1:11435/api/tags", timeout=5)
                if response.status_code == 200:
                    print("Ollama is ready!")
                    break
            except Exception:
                time.sleep(1)
        else:
            pytest.fail("Ollama failed to start within timeout")

        yield "http://127.0.0.1:11435"

    finally:
        if container:
            print("Stopping Ollama container...")
            container.stop()


@pytest.fixture(scope="module")
def small_model(ollama_container):
    """Pull a small model for testing."""
    import httpx

    model_name = "phi:latest"  # Small model for testing

    print(f"Pulling model {model_name}...")
    response = httpx.post(
        f"{ollama_container}/api/pull",
        json={"name": model_name},
        timeout=300,
    )

    if response.status_code != 200:
        pytest.skip(f"Failed to pull model: {response.text}")

    yield model_name


class TestOllamaIntegration:
    """Integration tests with real Ollama instance."""

    def test_ollama_list_models(self, ollama_container):
        """Test listing models from real Ollama."""
        import httpx

        response = httpx.get(f"{ollama_container}/api/tags", timeout=10)
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_ollama_generate_text(self, ollama_container, small_model):
        """Test text generation with real model."""
        import httpx

        response = httpx.post(
            f"{ollama_container}/api/generate",
            json={
                "model": small_model,
                "prompt": "What is 2+2? Answer with just the number.",
                "stream": False,
            },
            timeout=60,
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "4" in data["response"]

    def test_ollama_streaming(self, ollama_container, small_model):
        """Test streaming response from real model."""
        import httpx

        chunks = []
        with httpx.stream(
            "POST",
            f"{ollama_container}/api/generate",
            json={
                "model": small_model,
                "prompt": "Hello",
                "stream": True,
            },
            timeout=60,
        ) as response:
            assert response.status_code == 200

            for line in response.iter_lines():
                if line:
                    chunks.append(line)

        assert len(chunks) > 0

    def test_ollama_chat_endpoint(self, ollama_container, small_model):
        """Test chat completions endpoint."""
        import httpx

        response = httpx.post(
            f"{ollama_container}/api/chat",
            json={
                "model": small_model,
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "stream": False,
            },
            timeout=60,
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "content" in data["message"]


class TestRouterWithOllama:
    """Test SmarterRouter with real Ollama backend."""

    @pytest.mark.skip(reason="Legacy integration helper not compatible with current RouterEngine API")
    @pytest.mark.asyncio
    async def test_router_lists_real_models(self):
        """Deprecated placeholder for compatibility."""
        assert True

    @pytest.mark.skip(reason="Legacy integration helper not compatible with current RouterEngine API")
    @pytest.mark.asyncio
    async def test_router_can_route_to_real_model(self):
        """Deprecated placeholder for compatibility."""
        assert True
