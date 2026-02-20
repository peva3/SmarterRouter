from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # New capabilities
    vision: Mapped[bool] = mapped_column(Integer, default=0)  # SQLite uses Integer for Boolean
    tool_calling: Mapped[bool] = mapped_column(Integer, default=0)

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
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
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
    vision: Mapped[bool] = mapped_column(Integer, default=0)
    tool_calling: Mapped[bool] = mapped_column(Integer, default=0)

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
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    model_name: Mapped[str] = mapped_column(String, index=True)
    prompt_hash: Mapped[str | None] = mapped_column(String, index=True)

    # Feedback type: "positive" (1) or "negative" (-1)
    # Or detailed: score 1-5
    score: Mapped[float] = mapped_column(Float)  # 1.0 = good, 0.0 = bad, or -1.0 for dislike

    category: Mapped[str | None] = mapped_column(String)  # e.g. "coding", "reasoning"
    comment: Mapped[str | None] = mapped_column(String, nullable=True)
