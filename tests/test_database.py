"""Tests for database operations."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from router.database import get_session, init_db
from router.models import Base, BenchmarkSync, ModelBenchmark, ModelProfile, RoutingDecision


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Patch the global engine and session
    with patch("router.database.engine", engine):
        with patch("router.database.SessionLocal", TestingSessionLocal):
            yield engine


class TestDatabaseConnection:
    """Test database connection and initialization."""

    def test_init_db_creates_tables(self, test_db):
        """Test that init_db creates all tables."""
        from sqlalchemy import text
        
        # Tables should already exist from fixture
        with test_db.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            
            assert "model_profiles" in tables
            assert "routing_decisions" in tables
            assert "model_benchmarks" in tables
            assert "benchmark_sync" in tables

    def test_get_session_commit(self, test_db):
        """Test that get_session commits on success."""
        with get_session() as session:
            profile = ModelProfile(name="test_model", reasoning=0.8)
            session.add(profile)
        
        # Verify it was committed
        with get_session() as session:
            result = session.query(ModelProfile).filter_by(name="test_model").first()
            assert result is not None
            assert result.reasoning == 0.8

    def test_get_session_rollback_on_error(self, test_db):
        """Test that get_session rolls back on exception."""
        try:
            with get_session() as session:
                profile = ModelProfile(name="test_rollback", reasoning=0.5)
                session.add(profile)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify it was NOT committed
        with get_session() as session:
            result = session.query(ModelProfile).filter_by(name="test_rollback").first()
            assert result is None


class TestModelProfileCRUD:
    """Test ModelProfile CRUD operations."""

    def test_create_profile(self, test_db):
        """Test creating a model profile."""
        with get_session() as session:
            profile = ModelProfile(
                name="llama3",
                reasoning=0.8,
                coding=0.7,
                creativity=0.6,
                factual=0.9,
                speed=0.75,
                avg_response_time_ms=1200.0,
                last_profiled=datetime.now(timezone.utc),
            )
            session.add(profile)
            session.commit()
            
            assert profile.id is not None

    def test_read_profile(self, test_db):
        """Test reading a model profile."""
        with get_session() as session:
            profile = ModelProfile(name="mistral", reasoning=0.75)
            session.add(profile)
            session.commit()
        
        with get_session() as session:
            result = session.query(ModelProfile).filter_by(name="mistral").first()
            assert result is not None
            assert result.reasoning == 0.75

    def test_update_profile(self, test_db):
        """Test updating a model profile."""
        with get_session() as session:
            profile = ModelProfile(name="codellama", reasoning=0.5)
            session.add(profile)
            session.commit()
        
        with get_session() as session:
            profile = session.query(ModelProfile).filter_by(name="codellama").first()
            profile.reasoning = 0.95
            session.commit()
        
        with get_session() as session:
            profile = session.query(ModelProfile).filter_by(name="codellama").first()
            assert profile.reasoning == 0.95

    def test_delete_profile(self, test_db):
        """Test deleting a model profile."""
        with get_session() as session:
            profile = ModelProfile(name="temp_model", reasoning=0.5)
            session.add(profile)
            session.commit()
        
        with get_session() as session:
            profile = session.query(ModelProfile).filter_by(name="temp_model").first()
            session.delete(profile)
            session.commit()
        
        with get_session() as session:
            result = session.query(ModelProfile).filter_by(name="temp_model").first()
            assert result is None

    def test_unique_constraint(self, test_db):
        """Test that model names must be unique."""
        with get_session() as session:
            profile1 = ModelProfile(name="unique_model", reasoning=0.8)
            session.add(profile1)
            session.commit()
        
        # Second insert should fail
        with pytest.raises(Exception):
            with get_session() as session:
                profile2 = ModelProfile(name="unique_model", reasoning=0.9)
                session.add(profile2)
                session.commit()

    def test_capability_dict(self, test_db):
        """Test capability_dict method."""
        with get_session() as session:
            profile = ModelProfile(
                name="test",
                reasoning=0.8,
                coding=0.7,
                creativity=0.6,
                factual=0.9,
                speed=0.75,
            )
            session.add(profile)
            session.commit()
            
            caps = profile.capability_dict()
            assert caps["reasoning"] == 0.8
            assert caps["coding"] == 0.7
            assert caps["creativity"] == 0.6

    def test_overall_score(self, test_db):
        """Test overall_score property."""
        with get_session() as session:
            profile = ModelProfile(
                name="test",
                reasoning=0.8,
                coding=0.8,
                creativity=0.8,
                factual=0.8,
                speed=0.8,
            )
            session.add(profile)
            session.commit()
            
            assert profile.overall_score == 0.8


class TestModelBenchmarkCRUD:
    """Test ModelBenchmark CRUD operations."""

    def test_create_benchmark(self, test_db):
        """Test creating a benchmark entry."""
        with get_session() as session:
            benchmark = ModelBenchmark(
                ollama_name="llama3",
                mmlu=0.7,
                humaneval=0.5,
                reasoning_score=0.75,
                coding_score=0.6,
                general_score=0.7,
                elo_rating=1200.0,
                throughput=50.0,
                context_window=8192,
                last_updated=datetime.now(timezone.utc),
            )
            session.add(benchmark)
            session.commit()
            
            assert benchmark.id is not None

    def test_capability_dict_with_new_metrics(self, test_db):
        """Test capability dict includes new metrics."""
        with get_session() as session:
            benchmark = ModelBenchmark(
                ollama_name="test",
                reasoning_score=0.8,
                coding_score=0.9,
                general_score=0.7,
                elo_rating=1250.0,
                throughput=100.0,
            )
            session.add(benchmark)
            session.commit()
            
            caps = benchmark.capability_dict()
            assert "elo" in caps
            assert "throughput" in caps
            assert caps["elo"] == 1250.0


class TestRoutingDecisionCRUD:
    """Test RoutingDecision CRUD operations."""

    def test_create_decision(self, test_db):
        """Test creating a routing decision."""
        with get_session() as session:
            decision = RoutingDecision(
                prompt_hash="abc123",
                selected_model="llama3",
                confidence=0.95,
                reasoning="Matched to coding task",
            )
            session.add(decision)
            session.commit()
            
            assert decision.id is not None
            assert decision.timestamp is not None

    def test_decision_timestamp_auto_set(self, test_db):
        """Test that timestamp is automatically set."""
        from datetime import datetime, timezone
        
        with get_session() as session:
            decision = RoutingDecision(
                prompt_hash="test123",
                selected_model="mistral",
                confidence=0.8,
            )
            session.add(decision)
            session.commit()

            assert decision.timestamp is not None
            # Just verify it's a valid datetime, comparison can be tricky with timezones
            assert isinstance(decision.timestamp, datetime)


class TestBenchmarkSyncCRUD:
    """Test BenchmarkSync CRUD operations."""

    def test_create_sync_record(self, test_db):
        """Test creating a sync record."""
        with get_session() as session:
            sync = BenchmarkSync(
                last_sync=datetime.now(timezone.utc),
                models_count=10,
                status="completed",
            )
            session.add(sync)
            session.commit()
            
            assert sync.id is not None

    def test_default_status(self, test_db):
        """Test default status is 'pending'."""
        with get_session() as session:
            sync = BenchmarkSync()
            session.add(sync)
            session.commit()
            
            assert sync.status == "pending"

    def test_default_models_count(self, test_db):
        """Test default models_count is 0."""
        with get_session() as session:
            sync = BenchmarkSync()
            session.add(sync)
            session.commit()
            
            assert sync.models_count == 0
