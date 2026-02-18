"""Contract tests ensuring all backends behave consistently."""

import pytest
import httpx
import respx

from router.backends import create_backend
from router.backends.base import LLMBackend, ModelInfo, supports_unload
from router.config import Settings


class TestBackendContract:
    """Ensure all backends implement the contract consistently."""

    @pytest.fixture
    def ollama_settings(self):
        """Settings for Ollama backend."""
        return Settings(
            provider="ollama",
            ollama_url="http://mock-ollama:11434",
        )

    @pytest.fixture
    def llama_cpp_settings(self):
        """Settings for llama.cpp backend."""
        return Settings(
            provider="llama.cpp",
            llama_cpp_url="http://mock-llamacpp:8080",
        )

    @pytest.fixture
    def openai_settings(self):
        """Settings for OpenAI backend."""
        return Settings(
            provider="openai",
            openai_base_url="http://mock-openai:8000/v1",
            openai_api_key="sk-test",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["ollama_settings", "llama_cpp_settings", "openai_settings"])
    async def test_all_backends_list_models_returns_list(self, backend_fixture, request):
        """All backends must return list[ModelInfo] from list_models()."""
        settings = request.getfixturevalue(backend_fixture)
        backend = create_backend(settings)
        
        with respx.mock() as mock_http:
            if settings.provider == "ollama":
                mock_http.get("http://mock-ollama:11434/api/tags").mock(
                    return_value=httpx.Response(200, json={"models": []})
                )
            elif settings.provider == "llama.cpp":
                mock_http.get("http://mock-llamacpp:8080/v1/models").mock(
                    return_value=httpx.Response(200, json={"data": []})
                )
            else:  # openai
                mock_http.get("http://mock-openai:8000/v1/models").mock(
                    return_value=httpx.Response(200, json={"data": []})
                )
            
            models = await backend.list_models()
            assert isinstance(models, list)
            for m in models:
                assert isinstance(m, ModelInfo)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["ollama_settings", "llama_cpp_settings", "openai_settings"])
    async def test_all_backends_chat_returns_dict_with_message(self, backend_fixture, request):
        """All backends must return dict with 'message' key containing 'content'."""
        settings = request.getfixturevalue(backend_fixture)
        backend = create_backend(settings)
        
        with respx.mock() as mock_http:
            if settings.provider == "ollama":
                mock_http.post("http://mock-ollama:11434/api/chat").mock(
                    return_value=httpx.Response(200, json={
                        "message": {"content": "Test"},
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    })
                )
            elif settings.provider == "llama.cpp":
                mock_http.post("http://mock-llamacpp:8080/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "choices": [{"message": {"content": "Test"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                    })
                )
            else:  # openai
                mock_http.post("http://mock-openai:8000/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "choices": [{"message": {"content": "Test"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                    })
                )
            
            result = await backend.chat("test-model", [{"role": "user", "content": "Hello"}])
            
            assert isinstance(result, dict)
            assert "message" in result
            assert "content" in result["message"]
            assert result["message"]["content"] == "Test"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["ollama_settings", "llama_cpp_settings", "openai_settings"])
    async def test_all_backends_chat_includes_token_counts(self, backend_fixture, request):
        """All backends must include prompt_eval_count and eval_count in result."""
        settings = request.getfixturevalue(backend_fixture)
        backend = create_backend(settings)
        
        with respx.mock() as mock_http:
            if settings.provider == "ollama":
                mock_http.post("http://mock-ollama:11434/api/chat").mock(
                    return_value=httpx.Response(200, json={
                        "message": {"content": "Test"},
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    })
                )
            elif settings.provider == "llama.cpp":
                mock_http.post("http://mock-llamacpp:8080/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "choices": [{"message": {"content": "Test"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                    })
                )
            else:  # openai
                mock_http.post("http://mock-openai:8000/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "choices": [{"message": {"content": "Test"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                    })
                )
            
            result = await backend.chat("test-model", [{"role": "user", "content": "Hello"}])
            
            assert "prompt_eval_count" in result
            assert "eval_count" in result
            assert result["prompt_eval_count"] == 10
            assert result["eval_count"] == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["llama_cpp_settings", "openai_settings"])
    async def test_openai_backends_model_prefix_applied(self, backend_fixture, request):
        """Llama.cpp and OpenAI backends must apply model_prefix when configured."""
        settings = request.getfixturevalue(backend_fixture)
        settings.model_prefix = "prefixed/"
        backend = create_backend(settings)
        
        assert backend.model_prefix == "prefixed/"
        
        with respx.mock() as mock_http:
            if settings.provider == "llama.cpp":
                mock_http.post("http://mock-llamacpp:8080/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "choices": [{"message": {"content": "OK"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                    })
                )
            else:
                mock_http.post("http://mock-openai:8000/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "choices": [{"message": {"content": "OK"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                    })
                )
            
            await backend.chat("mymodel", [{"role": "user", "content": "test"}])
            
            import json
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["model"] == "prefixed/mymodel"

    @pytest.mark.asyncio
    async def test_ollama_backend_no_model_prefix_by_default(self, ollama_settings):
        """OllamaBackend doesn't apply prefix unless configured (default: no prefix)."""
        backend = create_backend(ollama_settings)
        assert backend.model_prefix == ""
        
        with respx.mock() as mock_http:
            mock_http.post("http://mock-ollama:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "OK"}})
            )
            await backend.chat("llama3", [{"role": "user", "content": "test"}])
            
            import json
            request = mock_http.calls.last.request
            body = json.loads(request.content)
            assert body["model"] == "llama3"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["ollama_settings", "llama_cpp_settings", "openai_settings"])
    async def test_all_backends_embed_returns_dict(self, backend_fixture, request):
        """All backends that support embed must return dict with embeddings."""
        settings = request.getfixturevalue(backend_fixture)
        backend = create_backend(settings)
        
        with respx.mock() as mock_http:
            if settings.provider == "ollama":
                mock_http.post("http://mock-ollama:11434/api/embed").mock(
                    return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
                )
            elif settings.provider == "llama.cpp":
                mock_http.post("http://mock-llamacpp:8080/v1/embeddings").mock(
                    return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
                )
            else:  # openai
                mock_http.post("http://mock-openai:8000/v1/embeddings").mock(
                    return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
                )
            
            result = await backend.embed("test-model", "text")
            
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["ollama_settings", "llama_cpp_settings", "openai_settings"])
    async def test_backends_handle_list_models_error(self, backend_fixture, request):
        """All backends must gracefully handle errors in list_models and return []."""
        settings = request.getfixturevalue(backend_fixture)
        backend = create_backend(settings)
        
        with respx.mock() as mock_http:
            if settings.provider == "ollama":
                mock_http.get("http://mock-ollama:11434/api/tags").mock(
                    side_effect=httpx.HTTPError("Connection failed")
                )
            elif settings.provider == "llama.cpp":
                mock_http.get("http://mock-llamacpp:8080/v1/models").mock(
                    side_effect=httpx.HTTPError("Connection failed")
                )
            else:
                mock_http.get("http://mock-openai:8000/v1/models").mock(
                    side_effect=httpx.HTTPError("Connection failed")
                )
            
            models = await backend.list_models()
            assert models == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_fixture", ["ollama_settings", "llama_cpp_settings", "openai_settings"])
    async def test_supports_unload_for_ollama_only(self, backend_fixture, request):
        """Only Ollama backend should report unload support."""
        settings = request.getfixturevalue(backend_fixture)
        backend = create_backend(settings)
        
        if settings.provider == "ollama":
            assert supports_unload(backend) is True
        else:
            assert supports_unload(backend) is False

    @pytest.mark.asyncio
    async def test_ollama_chat_streaming_format(self, ollama_settings):
        """Ollama streaming should yield Ollama-format chunks."""
        backend = create_backend(ollama_settings)
        
        stream_data = [
            b'{"message": {"content": "Hello"}}\n',
            b'{"message": {"content": " World"}, "done": false}\n',
            b'{"done": true}\n',
        ]

        with respx.mock() as mock_http:
            mock_http.post("http://mock-ollama:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b"".join(stream_data),
                    headers={"content-type": "text/plain"},
                )
            )
            stream, latency = await backend.chat_streaming("llama3", [{"role": "user", "content": "Hi"}])
            
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            
            assert any("message" in c for c in chunks)
