# Load testing with Locust
"""
Item #52: Load testing suite using Locust.

Simulates concurrent requests to measure performance under load.

Usage:
    cd tests/load
    locust -f locustfile.py --host=http://localhost:11436

Or headless mode:
    locust -f locustfile.py --host=http://localhost:11436 \
        --headless -u 100 -r 10 --run-time 5m
"""
import random
from typing import TYPE_CHECKING

from locust import HttpUser, between, task

if TYPE_CHECKING:
    from locust.clients import HttpSession


class ChatUser(HttpUser):
    """Simulates a user making chat completion requests."""

    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    host = "http://localhost:11436"

    def on_start(self):
        """Set up user session."""
        # Check health endpoint
        self.client.get("/health")

    @task(10)
    def chat_completion_simple(self):
        """Send simple chat completion requests."""
        prompts = [
            "What is 2+2?",
            "Explain quantum computing",
            "Write a Python hello world",
            "What is the capital of France?",
            "How does photosynthesis work?",
        ]

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [
                    {"role": "user", "content": random.choice(prompts)}
                ],
                "stream": False,
            },
            name="/v1/chat/completions [simple]",
            timeout=60,
        )

    @task(5)
    def chat_completion_complex(self):
        """Send complex chat completion requests."""
        prompts = [
            "Write a Python function to implement quicksort with documentation",
            "Explain the differences between REST and GraphQL APIs",
            "Generate a SQL query to find the top 10 customers by revenue",
            "Create a React component for a todo list with TypeScript",
            "Describe the architecture of a distributed message queue",
        ]

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [
                    {"role": "user", "content": random.choice(prompts)}
                ],
                "stream": False,
            },
            name="/v1/chat/completions [complex]",
            timeout=120,
        )

    @task(3)
    def chat_completion_streaming(self):
        """Send streaming chat completion requests."""
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [
                    {"role": "user", "content": "Write a short poem about AI"}
                ],
                "stream": True,
            },
            name="/v1/chat/completions [streaming]",
            timeout=60,
            stream=True,
        )

    @task(2)
    def list_models(self):
        """List available models."""
        self.client.get("/v1/models", name="/v1/models")

    @task(1)
    def health_check(self):
        """Check health endpoint."""
        self.client.get("/health", name="/health")


class AdminUser(HttpUser):
    """Simulates admin user making management requests."""

    wait_time = between(10, 30)  # Less frequent admin requests
    host = "http://localhost:11436"

    def on_start(self):
        """Set up admin session with API key."""
        # In real usage, set this from environment or config
        self.api_key = "admin-api-key"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @task(5)
    def get_profiles(self):
        """Get model profiles."""
        self.client.get(
            "/admin/profiles",
            headers=self.headers,
            name="/admin/profiles",
        )

    @task(3)
    def get_benchmarks(self):
        """Get benchmarks."""
        self.client.get(
            "/admin/benchmarks",
            headers=self.headers,
            name="/admin/benchmarks",
        )

    @task(2)
    def get_stats(self):
        """Get system stats."""
        self.client.get(
            "/admin/stats",
            headers=self.headers,
            name="/admin/stats",
        )

    @task(1)
    def get_health_detailed(self):
        """Get detailed health status."""
        self.client.get(
            "/health",
            name="/health [detailed]",
        )


class EmbeddingUser(HttpUser):
    """Simulates user making embedding requests."""

    wait_time = between(2, 8)
    host = "http://localhost:11436"

    @task(1)
    def create_embedding(self):
        """Create embeddings."""
        texts = [
            "Hello world",
            "Machine learning is fascinating",
            "Python is a great programming language",
            "The quick brown fox jumps over the lazy dog",
        ]

        self.client.post(
            "/v1/embeddings",
            json={
                "input": random.choice(texts),
                "model": "",
            },
            name="/v1/embeddings",
            timeout=30,
        )


class StressTestUser(HttpUser):
    """Heavy load user for stress testing."""

    wait_time = between(0.1, 0.5)  # Very short wait times
    host = "http://localhost:11436"

    @task(1)
    def rapid_requests(self):
        """Send rapid requests to stress test."""
        self.client.get("/health", name="/health [stress]")

    @task(1)
    def rapid_chat(self):
        """Send rapid chat requests."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
            name="/v1/chat/completions [stress]",
            timeout=10,
        )


def main():
    """Run load test from command line."""
    import sys

    print("Load testing configuration:")
    print("  ChatUser: Regular chat completion requests")
    print("  AdminUser: Admin API requests")
    print("  EmbeddingUser: Embedding requests")
    print("  StressTestUser: High-frequency stress test")
    print()
    print("Usage:")
    print("  locust -f locustfile.py --host=http://localhost:11436")
    print()
    print("Headless mode:")
    print("  locust -f locustfile.py --host=http://localhost:11436 \\")
    print("    --headless -u 100 -r 10 --run-time 5m")


if __name__ == "__main__":
    main()
