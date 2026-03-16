from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

if TYPE_CHECKING:
    from router.router import RoutingResult


class Base(DeclarativeBase):
    pass


class ModelProfile(Base):
    __tablename__ = "model_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)

    reasoning: Mapped[float] = mapped_column(Float, default=0.0)
    coding: Mapped[float] = mapped_column(Float, default=0.0)
    creativity: Mapped[float] = mapped_column(Float, default=0.0)
    factual: Mapped[float] = mapped_column(Float, default=0.0)
    speed: Mapped[float] = mapped_column(Float, default=0.0)

    avg_response_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    last_profiled: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    # Model availability tracking (SmarterRouter 2.1.6+)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # New capabilities
    vision: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_calling: Mapped[bool] = mapped_column(Boolean, default=False)

    # VRAM tracking (filled during profiling)
    vram_required_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    vram_measured_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    vram_quantization: Mapped[str | None] = mapped_column(String, nullable=True)

    # Profiling metadata
    adaptive_timeout_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    profiling_token_rate: Mapped[float | None] = mapped_column(Float, nullable=True)  # tokens/sec

    def capability_dict(self) -> dict[str, float]:
        return {
            "reasoning": self.reasoning,
            "coding": self.coding,
            "creativity": self.creativity,
            "factual": self.factual,
            "speed": self.speed,
        }

    @property
    def overall_score(self) -> float:
        caps = self.capability_dict()
        return sum(caps.values()) / len(caps) if caps else 0.0


class RoutingDecision(Base):
    __tablename__ = "routing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    response_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    prompt_hash: Mapped[str] = mapped_column(String, index=True)
    selected_model: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning: Mapped[str | None] = mapped_column(String, nullable=True)


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ollama_name: Mapped[str] = mapped_column(String, unique=True, index=True)

    mmlu: Mapped[float | None] = mapped_column(Float, nullable=True)
    humaneval: Mapped[float | None] = mapped_column(Float, nullable=True)
    math: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpqa: Mapped[float | None] = mapped_column(Float, nullable=True)
    hellaswag: Mapped[float | None] = mapped_column(Float, nullable=True)
    winogrande: Mapped[float | None] = mapped_column(Float, nullable=True)
    truthfulqa: Mapped[float | None] = mapped_column(Float, nullable=True)
    mmlu_pro: Mapped[float | None] = mapped_column(Float, nullable=True)

    reasoning_score: Mapped[float] = mapped_column(Float, default=0.0)
    coding_score: Mapped[float] = mapped_column(Float, default=0.0)
    general_score: Mapped[float] = mapped_column(Float, default=0.0)

    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    parameters: Mapped[str | None] = mapped_column(String, nullable=True)
    quantization: Mapped[str | None] = mapped_column(String, nullable=True)

    # New metrics
    elo_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    throughput: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # New capabilities
    vision: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_calling: Mapped[bool] = mapped_column(Boolean, default=False)

    # Extra provider-specific data (e.g., ArtificialAnalysis indices, speed metrics)
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def capability_dict(self) -> dict[str, float]:
        return {
            "reasoning": self.reasoning_score,
            "coding": self.coding_score,
            "general": self.general_score,
            "elo": self.elo_rating or 0.0,
            "throughput": self.throughput or 0.0,
        }

    @property
    def overall_score(self) -> float:
        caps = self.capability_dict()
        return sum(caps.values()) / len(caps) if caps else 0.0


class BenchmarkSync(Base):
    __tablename__ = "benchmark_sync"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_sync: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    models_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="pending")


class ModelFeedback(Base):
    """User feedback for model performance."""

    __tablename__ = "model_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    model_name: Mapped[str] = mapped_column(String, index=True)
    prompt_hash: Mapped[str | None] = mapped_column(String, index=True)

    # Feedback type: "positive" (1) or "negative" (-1)
    # Or detailed: score 1-5
    score: Mapped[float] = mapped_column(Float)  # 1.0 = good, 0.0 = bad, or -1.0 for dislike

    category: Mapped[str | None] = mapped_column(String)  # e.g. "coding", "reasoning"
    comment: Mapped[str | None] = mapped_column(String, nullable=True)


class RoutingCache(Base):
    """Persistent cache for routing decisions."""

    __tablename__ = "routing_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cache_key: Mapped[str] = mapped_column(String, unique=True, index=True)
    selected_model: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_magnitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    access_count: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_routing_result(self) -> RoutingResult:  # type: ignore[name-defined]
        """Convert to RoutingResult dataclass."""
        from router.router import RoutingResult

        return RoutingResult(
            selected_model=self.selected_model,
            confidence=self.confidence,
            reasoning=self.reasoning or "",
        )


class ResponseCache(Base):
    """Persistent cache for LLM responses."""

    __tablename__ = "response_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, index=True)
    prompt_hash: Mapped[str] = mapped_column(String, index=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_text: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    access_count: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def cache_key(self) -> tuple:
        """Generate cache key matching SemanticCache._make_response_key."""
        if self.parameters:
            param_tuple = tuple(sorted((k, v) for k, v in self.parameters.items() if v is not None))
            return (self.model_name, self.prompt_hash, param_tuple)
        return (self.model_name, self.prompt_hash)


class EmbeddingCache(Base):
    """Persistent cache for embeddings."""

    __tablename__ = "embedding_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    embedding: Mapped[list[float]] = mapped_column(JSON)
    magnitude: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    access_count: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AdminAuditLog(Base):
    """Audit log for admin actions."""

    __tablename__ = "admin_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )

    action: Mapped[str] = mapped_column(String, index=True)  # e.g. "reprofile", "cache_clear"
    endpoint: Mapped[str] = mapped_column(String)  # e.g. "/admin/reprofile"
    method: Mapped[str] = mapped_column(String)  # HTTP method: GET, POST, etc.

    ip_address: Mapped[str | None] = mapped_column(String, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String, nullable=True)

    # Request details (sanitized)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Response summary
    result_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, default=200)

    # Duration in milliseconds
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class BackgroundTaskDLQ(Base):
    """Dead letter queue for failed background tasks."""

    __tablename__ = "background_task_dlq"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_name: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True, default="failed")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
