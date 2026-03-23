"""Test for the Ollama backend multimodal fix."""
import json

import httpx
import pytest
import respx

from router.backends.ollama import OllamaBackend


@pytest.mark.asyncio
async def test_ollama_multimodal_fix_base64_image():
    """Test that base64 images are properly transformed for Ollama."""
    backend = OllamaBackend("http://localhost:11434")

    with respx.mock() as mock_http:
        mock_http.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "Image described"}})
        )
        
        # Simulate Open WebUI sending a base64 image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}},
                ]
            }
        ]
        
        await backend.chat("llava", messages, stream=False)
        
        # Verify the transformation
        request = mock_http.calls.last.request
        body = json.loads(request.content)
        
        assert len(body["messages"]) == 1
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "What's in this image?"
        assert "images" in msg
        assert len(msg["images"]) == 1
        # Verify base64 prefix was stripped
        assert msg["images"][0] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.mark.asyncio
async def test_ollama_multimodal_fix_regular_url():
    """Test that regular image URLs are preserved."""
    backend = OllamaBackend("http://localhost:11434")

    with respx.mock() as mock_http:
        mock_http.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "Image described"}})
        )
        
        # Simulate Open WebUI sending a regular URL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                ]
            }
        ]
        
        await backend.chat("llava", messages, stream=False)
        
        # Verify the transformation
        request = mock_http.calls.last.request
        body = json.loads(request.content)
        
        assert len(body["messages"]) == 1
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Describe this image"
        assert "images" in msg
        assert len(msg["images"]) == 1
        # Verify URL was preserved as-is
        assert msg["images"][0] == "http://example.com/image.jpg"


@pytest.mark.asyncio
async def test_ollama_multimodal_fix_multiple_images():
    """Test multiple images in a single message."""
    backend = OllamaBackend("http://localhost:11434")

    with respx.mock() as mock_http:
        mock_http.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "Two images described"}})
        )
        
        # Multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,first"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,second"}},
                ]
            }
        ]
        
        await backend.chat("llava", messages, stream=False)
        
        # Verify the transformation
        request = mock_http.calls.last.request
        body = json.loads(request.content)
        
        assert len(body["messages"]) == 1
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Compare these:"
        assert "images" in msg
        assert len(msg["images"]) == 2
        assert msg["images"][0] == "first"
        assert msg["images"][1] == "second"


@pytest.mark.asyncio
async def test_ollama_multimodal_fix_streaming():
    """Test that streaming also works with multimodal content."""
    backend = OllamaBackend("http://localhost:11434")

    with respx.mock() as mock_http:
        mock_http.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                content=b'{"message": {"content": "Streaming image desc"}}\n{"done": true}\n',
                headers={"content-type": "text/plain"},
            )
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Stream this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,streamtest"}},
                ]
            }
        ]
        
        stream, latency = await backend.chat_streaming("llava", messages)
        
        # Consume the stream
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Verify the transformation
        request = mock_http.calls.last.request
        body = json.loads(request.content)
        
        assert len(body["messages"]) == 1
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Stream this image"
        assert "images" in msg
        assert len(msg["images"]) == 1
        assert msg["images"][0] == "streamtest"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])